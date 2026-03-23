# IMBE-ASR: Speech Recognition Directly from Vocoder Parameters

ASR straight from P25 IMBE codec parameters -- skip audio reconstruction entirely and go from the digital bitstream to text. 290M Conformer-CTC, **1.9% WER** with language model on LibriSpeech-IMBE.

**Models:** [trunk-reporter/imbe-asr collection](https://huggingface.co/collections/trunk-reporter/imbe-asr-speech-recognition-from-vocoder-parameters-69c0a4f68ef670b5bf68449d) on Hugging Face

| Model | Params | Greedy WER | Beam WER | HF Repo |
|-------|--------|-----------|----------|---------|
| Large | 290M | 6.5% | **1.9%** | [imbe-asr-large-1024d](https://huggingface.co/trunk-reporter/imbe-asr-large-1024d) |
| Base | 48.6M | 18.9% | -- | [imbe-asr-base-512d](https://huggingface.co/trunk-reporter/imbe-asr-base-512d) |
| Base P25 | 48.6M | 19.2% | -- | [imbe-asr-base-512d-p25](https://huggingface.co/trunk-reporter/imbe-asr-base-512d-p25) |

All models available in SafeTensors + ONNX (fp32, int8, uint8) formats.

## The Problem

P25 digital radio (law enforcement, fire, EMS) uses the IMBE vocoder -- a 4.4 kbps codec from the 90s. If you want to transcribe it, the standard approach is: decode IMBE -> reconstruct audio -> run Whisper or whatever.

But there is no public IMBE specification. The codec is proprietary (DVSI). `libimbe` -- the open-source implementation everyone uses -- is a reverse-engineered best guess at the original algorithm. The audio it produces is an approximation of an approximation. So the standard pipeline is:

**codec params -> proprietary-spec-approximated waveform reconstruction -> mel spectrogram extraction -> ASR**

Every step is lossy. The reconstruction itself is computationally expensive (trig, IFFT, overlap-add synthesis per frame), and at the end you've created a degraded audio signal so a speech model can re-extract spectral features -- features that were already there in the codec parameters.

We skip all of that.

## Why Skip Reconstruction

The vocoder parameters already encode speech: fundamental frequency, spectral amplitudes, voicing decisions. That IS the speech, in a compact parametric representation. The model doesn't need a waveform -- it needs phonetic information, which the codec parameters describe directly.

The computational argument: IMBE waveform synthesis requires per-frame harmonic oscillator evaluation, spectral amplitude interpolation, voiced/unvoiced mixing, and overlap-add windowing. Then you still need a spectrogram. Our approach is a single matrix multiply (170-dim -> 1024-dim) per frame. Full inference on a 10-second call: **5.8ms on GPU** (200x real-time). The vocoder reconstruction alone would take longer.

We also sidestep the fidelity question entirely. It doesn't matter how accurate libimbe's reconstruction is, because we never reconstruct. We go straight from the digital bitstream to text.

## Input Representation

Each 20ms IMBE frame is expanded into a 170-dimensional vector:

```
[0]       f0 (fundamental frequency)
[1]       L (harmonic count, 0-56)
[2:58]    spectral amplitudes (56 slots, zero-padded)
[58:114]  voiced/unvoiced flags per harmonic
[114:170] binary validity mask (1=real harmonic, 0=padding)
```

The mask is critical -- without it the model can't distinguish silence from zero-padded harmonics. We tried raw 8-dim codewords (too compressed) and 20-dim BFCCs (lost information). This 170-dim representation won decisively.

## Results

**LibriSpeech-IMBE validation** (speaker-split, 2775 utterances):

| Decoding | WER | CER |
|----------|-----|-----|
| Greedy | 6.5% | 1.9% |
| + 5-gram KenLM | **1.9%** | **0.7%** |

For context: this is from 4.4 kbps vocoder parameters, not audio. The IMBE codec was designed in 1993 to be barely intelligible to human ears.

Full eval logs with REF/HYP examples are in [`results/`](results/).

**Scaling progression:**

| Model | Params | Greedy WER |
|-------|--------|-----------|
| Proof of concept (d=256, 6L) | 9M | 40.6% |
| Base (d=512, 8L) | 48.6M | 18.9% |
| Large (d=1024, 12L) | 290M | 6.5% |

**Training curve** (290M model, 30 epochs on ~1220h IMBE-encoded speech):

```
Epoch  2: WER=55.2%  val_loss=0.873
Epoch 10: WER=25.4%  val_loss=0.443
Epoch 20: WER=16.4%  val_loss=0.350
Epoch 30: WER=14.7%  val_loss=0.379
```

Training WER is on the full multi-source val set (LibriSpeech + TEDLIUM + GigaSpeech). The 6.5%/1.9% numbers are on the LibriSpeech speaker-split val, comparable to standard benchmarks. Full training curve in [`results/training_curve_1024d.txt`](results/training_curve_1024d.txt).

W&B run: [sarah-1024d-12l-ddp](https://wandb.ai/luxprimatech/imbe-asr/runs/zrnez9mv)

**Inference speed (GPU workstation):**

| Call Duration | GPU (3090 Ti) | CPU |
|---------------|--------------|-----|
| 1s | 5.8ms | 12ms |
| 5s | 5.8ms | 25ms |
| 10s | 5.8ms | 49ms |
| 30s | 6.9ms | 241ms |

200x real-time on CPU, 1700x on GPU.

### Edge Deployment (Raspberry Pi 5, 4GB)

The 290M model runs on a $60 Raspberry Pi via ONNX Runtime + int8 quantization. PyTorch can't even load it (OOM at 3.3GB), but quantized to 298-312MB it fits comfortably.

| Runtime | Model | Size | 10s call | RTF | RAM |
|---------|-------|------|----------|-----|-----|
| C (70KB binary) | 48.6M fp32 | 195 MB | 660ms | 0.07x | ~300 MB |
| C (70KB binary) | 48.6M uint8 | 59 MB | 788ms | 0.08x | ~140 MB |
| C (70KB binary) | 290M uint8 | 312 MB | 2.8s | 0.28x | ~1.3 GB |
| Python ONNX RT | 48.6M fp32 | 195 MB | 649ms | 0.07x | 285 MB |
| Python ONNX RT | 290M int8 | 298 MB | 3.5s | 0.35x | 535 MB |
| PyTorch | 48.6M fp32 | 567 MB | 800ms | 0.08x | 995 MB |
| PyTorch | 290M fp32 | 3.3 GB | OOM | -- | >4 GB |

All models faster than real-time. The C engine is a 70KB binary -- no Python needed. int8 quantization preserves perfect accuracy on the 290M model.

Full benchmark logs in [`results/`](results/).

## Architecture

Conformer-CTC. Nothing exotic -- the interesting part is the input, not the model.

- Input projection: Linear(170, 1024) + LayerNorm + Dropout
- Encoder: 12 x ConformerBlock (FFN/2 -> MHSA -> ConvModule(k=31) -> FFN/2 -> LN)
- CTC head: Linear(1024, 40) -> log_softmax
- 290M parameters
- Character-level tokenizer: blank + A-Z + 0-9 + space + apostrophe = 40 classes

Trained with CTC loss, cosine LR schedule, AdamW, bf16 mixed precision on 2x RTX 3090 Ti (~4 days).

Bayesian hyperparameter sweep confirmed depth matters more than width. Optimal: d=1024, 12 layers, ff_mult=4.

## Training Data

~1220 hours of clean speech (LibriSpeech 960h, TEDLIUM 3 ~452h, GigaSpeech S ~250h), IMBE-encoded through libimbe to produce 170-dim frame parameters. The model learns to read IMBE parameters the way a conventional ASR model reads mel spectrograms.

Subtle detail: software-encoded IMBE has a 2-frame analysis delay (frame N describes audio at frame N-2). Real P25 radio doesn't have this. All software-encoded training data is trimmed accordingly.

For P25 domain adaptation: ~20 hours of real radio captures, pseudo-labeled with a Whisper large-v3 + Qwen3-ASR ensemble, filtered for hallucinations.

## Reproducing

```bash
# Precompute 170-dim IMBE features from audio
python -m src.precompute --pairs-dir data/pairs --workers 12

# Train (multi-source, DDP)
torchrun --nproc_per_node=2 -m src.train \
    --mmap-dir data/packed \
    --d-model 1024 --n-layers 12 --n-heads 16 --d-ff 4096 \
    --epochs 30 --batch-size 4 --accum-steps 32 --lr 3e-4

# Evaluate (greedy)
python -m src.eval checkpoints/best.pth \
    --pairs-dir data/pairs --batch-size 16

# Evaluate (beam search + LM)
python -m src.eval checkpoints/best.pth \
    --beam --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt

# Inference on a P25 .tap file
python -m src.inference --checkpoint checkpoints/best.pth \
    --tap-file path/to/call.tap

# Watch directory for live transcription
python -m src.inference --checkpoint checkpoints/best.pth \
    --watch ~/trunk-recorder/audio/

# P25 fine-tuning (DDP)
torchrun --nproc_per_node=2 scripts/finetune_p25.py \
    --checkpoint checkpoints/best.pth \
    --p25-dir data/p25_labeled --base-mmap data/packed \
    --epochs 15 --lr 3e-5 --amp
```

## Project Structure

```
src/
  model.py            Conformer-CTC architecture (scalable: 9M to 290M)
  tokenizer.py        Character-level CTC tokenizer (40 classes)
  dataset.py          LibriSpeech IMBE dataset + speaker splits
  dataset_unified.py  Multi-source dataset + memory-mapped format
  dataset_p25.py      Real P25 dataset (talkgroup splits)
  train.py            Training loop (CTC, cosine LR, DDP multi-GPU)
  eval.py             WER/CER evaluation (greedy + beam)
  decode.py           Beam search with KenLM
  inference.py        TAP file / watch mode / streaming inference
  symbolstream_client.py  Live transcription via symbolstream plugin (TCP)
  live.py             Legacy Unix socket transcription
  precompute.py       Audio -> 170-dim IMBE features via libimbe

scripts/
  finetune_p25.py     P25 fine-tuning with base data mixing (DDP)
  pseudo_label.py     Multi-ASR pseudo-labeling for P25
  prepare_*.py        Dataset preparation scripts
  build_lm.py         KenLM language model training

results/              Eval logs and training curves
configs/              Training configurations
```

## Getting IMBE Symbols from P25 Radio

**Important:** Standard trunk-recorder only outputs reconstructed audio. To use IMBE-ASR, you need the raw IMBE codec symbols (codewords) before audio reconstruction.

Our [fork of trunk-recorder](https://github.com/trunk-reporter/trunk-recorder) adds a `voice_codec_data()` callback that exposes the raw IMBE frame vectors. The [symbolstream](https://github.com/trunk-reporter/symbolstream) plugin uses this callback to stream raw codec symbols over TCP/UDP to a remote server -- the same way `simplestream` streams audio, but with the pre-vocoder codec parameters instead. This is what the model consumes: the 8 codeword parameters per 20ms frame, decoded into 170-dim features via libimbe.

Without this fork and plugin (or another source of raw IMBE symbols), the models cannot be used on live radio. The whole point is to skip audio reconstruction -- if you only have audio, use a conventional ASR model like [Whisper](https://github.com/openai/whisper) or our [Qwen3-ASR P25 fine-tune](https://huggingface.co/trunk-reporter/qwen3-asr-p25-0.6B).

### Live transcription with symbolstream

Start the symbolstream client (it listens for the plugin to connect):

```bash
python -m src.symbolstream_client \
    --checkpoint checkpoints/best.pth \
    --port 9090 --stream
```

Then configure the symbolstream plugin in trunk-recorder's `config.json` to point at this machine:

```json
{
    "name": "symbolstream",
    "library": "libsymbolstream_plugin",
    "streams": [{
        "address": "<this machine's IP>",
        "port": 9090,
        "TGID": 0,
        "useTCP": true,
        "sendJSON": true
    }]
}
```

## Dependencies

- Python 3.10+, PyTorch 2.0+
- `libimbe.so` -- IMBE vocoder C library (for feature extraction from codewords)
- [trunk-recorder fork](https://github.com/trunk-reporter/trunk-recorder) (for live P25 symbol capture)
- pyctcdecode + kenlm (beam search decoding)
- wandb (experiment tracking)

## Beyond IMBE

The approach is codec-agnostic. IMBE and AMBE/AMBE+2 are the same family (both DVSI) -- same parametric structure: f0, spectral amplitudes, voiced/unvoiced decisions. Swapping `libimbe` for `libambe` in feature extraction is the only change needed. Not yet tested, but if it transfers, the coverage is significant:

| Codec | Bitrate | Systems |
|-------|---------|---------|
| IMBE | 4.4 kbps | P25 Phase 1 |
| AMBE+2 | 2.4-9.6 kbps | P25 Phase 2, DMR, dPMR, NXDN, D-STAR, Fusion |
| Codec2 | 0.7-3.2 kbps | FreeDV, open-source HF radio |
| MELPe | 1.2-2.4 kbps | NATO STANAG 4591, military |

DMR alone covers most commercial/industrial two-way radio worldwide. D-STAR is amateur radio. NXDN is Kenwood/Icom. If the principle holds -- that speech codecs are already doing feature extraction and we just need to learn to read their output -- this generalizes well beyond P25.

## What's Next

- P25 fine-tuning the 290M model (in progress)
- Live transcription pipeline from trunk-recorder IMBE tap
- AMBE+2 support (DMR, P25 Phase 2, D-STAR)
