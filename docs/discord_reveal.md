# IMBE-ASR: Speech Recognition Directly from Vocoder Parameters

**tl;dr** -- skip audio reconstruction, do ASR straight from P25 IMBE codec parameters. 290M Conformer-CTC, **1.9% WER** with LM. runs on a Raspberry Pi.

P25 digital radio uses the IMBE vocoder -- a 4.4 kbps codec from the 90s. everyone transcribes it by reconstructing audio first, then running Whisper or whatever. but `libimbe` (the open-source implementation) is a reverse-engineered best guess at a proprietary codec. the audio it produces is an approximation of an approximation.

so we skipped all of that. the vocoder parameters already encode f0, spectral amplitudes, and voicing decisions -- that IS the speech. we trained a Conformer-CTC model to read 170-dim IMBE parameters directly. no waveform synthesis, no spectrogram extraction, no accumulated approximation error.

## results

```
decoding          | WER   | CER
------------------|-------|------
greedy            | 6.5%  | 1.9%
+ 5-gram KenLM   | 1.9%  | 0.7%
```

from 4.4 kbps vocoder parameters. not audio.

## it runs on a pi

```
runtime              | model      | 10s call | RTF
---------------------|------------|----------|--------
C engine (70KB bin)  | 290M uint8 | 2.8s     | 0.28x
C engine (70KB bin)  | 48.6M fp32 | 660ms    | 0.07x
PyTorch              | 290M fp32  | OOM      | --
```

the 290M model quantized to int8 (298MB) fits on a 4GB Raspberry Pi 5 where PyTorch can't even load it. perfect accuracy preserved. 70KB binary, no Python needed.

## links

- **code**: https://github.com/trunk-reporter/imbe-asr
- **models**: https://huggingface.co/collections/trunk-reporter/imbe-asr-speech-recognition-from-vocoder-parameters-69c0a4f68ef670b5bf68449d
- **best model** (290M, 1.9% WER): https://huggingface.co/trunk-reporter/imbe-asr-large-1024d
- **edge model** (48.6M, Pi-friendly): https://huggingface.co/trunk-reporter/imbe-asr-base-512d

P25 fine-tune of the 290M model training now.

## beyond IMBE

the approach is codec-agnostic. IMBE and AMBE/AMBE+2 are the same family (both DVSI) -- same parametric structure: f0, spectral amplitudes, voiced/unvoiced decisions. swapping `libimbe` for `libambe` in the feature extraction is the only change needed. this hasn't been tested yet, but if it transfers, the coverage is huge:

```
codec    | bitrate     | systems
---------|-------------|------------------------------------------
IMBE     | 4.4 kbps    | P25 Phase 1
AMBE+2   | 2.4-9.6 kbps| P25 Phase 2, DMR, dPMR, NXDN, D-STAR, Fusion
Codec2   | 0.7-3.2 kbps| FreeDV, open-source HF radio
MELPe    | 1.2-2.4 kbps| NATO STANAG 4591, military
```

DMR alone covers most commercial/industrial two-way radio worldwide. D-STAR is amateur radio. NXDN is Kenwood/Icom. if the principle holds -- that speech codecs are already doing feature extraction and we just need to learn to read their output -- this generalizes well beyond P25.
