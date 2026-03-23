# IMBE-ASR: Speech Recognition Directly from Vocoder Parameters

**tl;dr** -- we skip audio reconstruction entirely and do ASR straight from P25 IMBE codec parameters. 290M Conformer-CTC, 1.9% WER with LM on LibriSpeech-IMBE. it works way better than it should.

## the problem

P25 digital radio (law enforcement, fire, EMS) uses the IMBE vocoder -- a 4.4 kbps codec from the 90s that sounds like absolute garbage. if you want to transcribe it, the standard approach is: decode IMBE → reconstruct audio → run Whisper or whatever.

but here's the thing: there is no public IMBE specification. the codec is proprietary (DVSI). `libimbe` -- the open-source implementation everyone uses -- is a reverse-engineered best guess at the original algorithm. the audio it produces is an *approximation of an approximation*. so the standard pipeline is actually:

**codec params → proprietary-spec-approximated waveform reconstruction → mel spectrogram extraction → ASR**

every step is lossy. the reconstruction itself is computationally expensive (trig, IFFT, overlap-add synthesis per frame), and at the end of it you've just created a degraded audio signal so that a speech model can re-extract spectral features from it -- features that were *already there* in the codec parameters you started with.

so we asked: what if you just... don't reconstruct the audio?

## why skip reconstruction

the vocoder parameters already encode speech. fundamental frequency, spectral amplitudes, voicing decisions -- that IS the speech, just in a compact parametric representation. the model we're training doesn't need to hear a waveform. it needs to know what phonemes are being produced. and the codec parameters describe exactly that, before any lossy reconstruction muddies the water.

there's also a computational argument. IMBE waveform synthesis through libimbe requires per-frame: harmonic oscillator bank evaluation, spectral amplitude interpolation, voiced/unvoiced mixing, overlap-add windowing. then you still need to compute a spectrogram on the output. our approach is a single matrix multiply (170-dim → 1024-dim linear projection) per frame. the entire 10-second-call inference is 5.8ms on GPU -- the vocoder reconstruction alone would take longer than that.

we're also sidestepping the fidelity question entirely. it doesn't matter how accurate libimbe's reconstruction is, because we never reconstruct. we go straight from the digital bitstream to text. if the codec parameters faithfully represent what was spoken (and they do -- that's their job), then we have everything we need.

## the input representation

each 20ms IMBE frame decodes into 8 codeword parameters, which we expand through libimbe into a 170-dimensional vector:

```
[0]       f0 (fundamental frequency)
[1]       L (harmonic count, 0-56)
[2:58]    spectral amplitudes (56 slots, zero-padded)
[58:114]  voiced/unvoiced flags per harmonic
[114:170] binary validity mask (1=real harmonic, 0=padding)
```

the mask is critical. without it the model can't distinguish silence from zero-padded harmonics. we tried raw 8-dim codewords (too compressed, model couldn't learn), 20-dim BFCCs (lost information), and this 170-dim "middle path" representation. the middle path won decisively.

## architecture

Conformer-CTC. nothing exotic -- the interesting part is the input, not the model.

- input projection: Linear(170, 1024) + LayerNorm + Dropout
- encoder: 12 x ConformerBlock (FFN/2 -> MHSA -> ConvModule(k=31) -> FFN/2 -> LN)
- CTC head: Linear(1024, 40) -> log_softmax
- 290M parameters total
- character-level tokenizer: blank + A-Z + 0-9 + space + apostrophe = 40 classes

trained with CTC loss, cosine LR schedule, AdamW, bf16 mixed precision. 2x 3090 Ti for ~4 days.

## training data

we don't have 1000 hours of transcribed P25 radio. so we did something cursed: took clean speech datasets (LibriSpeech, TEDLIUM 3, GigaSpeech S -- ~1220 hours total), IMBE-encoded all of it through libimbe (audio -> vocoder params -> 170-dim features), and trained on that. the model learns to read IMBE parameters as if they were mel spectrograms.

one subtle gotcha: software-encoded IMBE has a 2-frame analysis delay (frame N describes audio at frame N-2). real P25 radio doesn't have this. we trim the first 2 frames from all software-encoded training data to keep alignment clean.

for P25 adaptation: ~20 hours of real radio captures, pseudo-labeled with a Whisper + Qwen3 ensemble, filtered for hallucinations.

## results

**LibriSpeech-IMBE validation (speaker-split, 2775 utterances):**

```
decoding          | WER   | CER
------------------|-------|------
greedy            | 6.5%  | 1.9%
+ 5-gram KenLM   | 1.9%  | 0.7%
```

for context: this is from 4.4 kbps vocoder parameters, not audio. the IMBE codec was designed in 1993 to be barely intelligible to human ears. getting 1.9% WER from it is... unexpected.

example output (beam search):
```
REF: AS IF THE YOUNG LADY HAD BEEN IN A POSITION TO APPEAL TO IT BUT IN FACT
     THE BRITISH PUBLIC REMAINED FOR THE PRESENT PROFOUNDLY INDIFFERENT
HYP: AS IF THE YOUNG LADY HAD BEEN IN A POSITION TO APPEAL TO IT BUT IN FACT
     THE BRITISH PUBLIC REMAINED FOR THE PRESENT PROFOUNDLY INDIFFERENT
```

basically perfect. residual errors are things like HALL->WHOLE (acoustically identical through IMBE) and WHAT'S->WHAT IS (contraction ambiguity).

**scaling progression:**

```
model                        | params | greedy WER
-----------------------------|--------|----------
proof of concept (d=256, 6L) | 9M     | 40.6%
base (d=512, 8L)             | 48.6M  | 18.9%
large (d=1024, 12L)          | 290M   | 6.5%
```

bayesian hyperparameter sweep confirmed: depth matters more than width, optimal config is d=1024, 12 layers, ff_mult=4.

## why this matters

1. **no audio reconstruction needed.** the entire vocoder -> audio -> ASR pipeline is replaced by a single forward pass on the raw codec parameters. no reverse-engineered waveform synthesis, no spectrogram extraction, no accumulated approximation error.

2. **works on any IMBE stream.** P25, DSTAR, any system using IMBE/AMBE vocoders. you can tap the digital bitstream before any audio is ever synthesized.

3. **the codec preserves more than we thought.** IMBE was designed for intelligibility at 4.4 kbps, not for downstream ML. but the spectral amplitudes + voicing decisions + f0 apparently encode enough phonetic information for near-perfect recognition. the model is learning something the codec designers never intended.

4. **generalizes to the actual application.** P25-finetuned model produces readable dispatch transcriptions from real radio traffic. addresses, unit numbers, natural conversation -- not just 10-codes.

## what's next

P25 fine-tuning the 290M model now (should finish overnight). after that: live transcription pipeline from trunk-recorder's IMBE tap, real-time dispatch transcription for the actual use case.

the broader question is whether this approach generalizes to other low-bitrate codecs. if IMBE works this well at 4.4 kbps, what about Codec2 at 700 bps? MELPe at 2.4 kbps? there might be a general principle here: speech codecs are already doing feature extraction, we just need to learn to read their output.
