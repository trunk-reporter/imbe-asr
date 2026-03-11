# IMBE-ASR: Speech Recognition Directly from P25 Vocoder Parameters

A speech-to-text system that reads IMBE vocoder parameters from P25 radio transmissions and produces text transcriptions -- **without ever synthesizing audio**. Instead of the traditional pipeline (radio -> IMBE decode -> audio -> ASR), we skip the audio entirely and go straight from codec parameters to text.

## Why

P25 (Project 25) is the digital radio standard used by public safety agencies across the US -- law enforcement, fire, EMS, public works, transit, schools, and more. Every P25 voice transmission is encoded with the IMBE (Improved Multi-Band Excitation) vocoder, which compresses speech into 8 codeword parameters per 20ms frame.

Existing transcription approaches decode these parameters back to audio first, then run a general-purpose ASR model (Whisper, etc.) on the reconstructed waveform. This is wasteful: the IMBE vocoder destroys phase information and limits bandwidth to ~3.4kHz, so the "audio" is a poor representation of what was actually said. The codec parameters themselves contain the same spectral information in a more direct form.

By training an ASR model that takes decoded IMBE parameters as input, we:

- **Skip lossy audio reconstruction** -- no vocoder synthesis artifacts
- **Preserve full spectral resolution** -- per-harmonic amplitudes and voicing
- **Run faster** -- 170 floats per frame vs 160 audio samples (at 8kHz)
- **Enable pre-audio-decode transcription** -- tap the codec bitstream before vocoder synthesis (via the IMBE tap socket in trunk-recorder)

## How It Works

### Input Representation (170 dimensions per frame)

Each 20ms IMBE frame carries 8 codeword parameters `u[0..7]`. We run these through the deterministic IMBE decode (the same math the vocoder uses) but stop before waveform synthesis:

```
u[0..7] -> imbe_decode_params() -> f0, L, spectral_amplitudes, voicing_flags
```

This produces a fixed-size 170-dimensional feature vector per frame:

| Dims | Content | Description |
|------|---------|-------------|
| `[0]` | f0 | Fundamental frequency (Hz) |
| `[1]` | L | Number of harmonics (0-56) |
| `[2:58]` | sa[0..55] | Spectral amplitudes (zero-padded) |
| `[58:114]` | vuv[0..55] | Per-harmonic voiced/unvoiced flags (zero-padded) |
| `[114:170]` | mask[0..55] | Binary harmonic validity mask (1=real, 0=pad) |

The mask is critical -- without it, the model can't distinguish zero-energy harmonics from padding, which matters because IMBE preserves per-band spectral granularity.

This is the **"middle path"** between:
- Raw codewords (8-dim, too compressed for the model to learn the decode math)
- BFCCs (20-dim, lossy DCT smoothing discards per-harmonic detail)

### Model Architecture

**Conformer-CTC** encoder with character-level CTC decoding:

```
170-dim IMBE params -> Linear(170, 256) -> 6x Conformer blocks -> CTC head -> 40 characters
```

Each Conformer block: FFN(1/2) -> Multi-Head Self-Attention -> ConvModule(k=31) -> FFN(1/2) -> LayerNorm

- 9.2M parameters (proof-of-concept size)
- 40-class character vocabulary: blank + A-Z + 0-9 + space + apostrophe
- Optional beam search with KenLM n-gram language model

### Data Sources

1. **LibriSpeech train-clean-100** (103.7 hours, 28,539 utterances) -- Clean audiobook speech IMBE-encoded via `generate_pairs.py` to produce paired (IMBE params, transcript) training data.

2. **Real P25 TAP recordings** (~37,800 calls) -- Live radio traffic captured via the IMBE tap socket in trunk-recorder. Transcribed by a P25-tuned Qwen3-ASR server. Covers law enforcement, fire, EMS, public works, transit, and general conversational radio traffic.

## Results

### LibriSpeech-IMBE validation (greedy decode, no LM)

| Epoch | WER | CER |
|-------|-----|-----|
| 10 | 65.8% | 26.6% |
| 20 | 48.7% | 18.1% |
| 30 | 42.4% | 15.3% |
| 36 | **40.6%** | **14.6%** |

With beam search + 5-gram LM: ~28-30% WER.

For comparison, the previous BFCC-based approach (20-dim input) plateaued at 56% greedy WER / 36% beam+LM WER. The 170-dim raw params are strictly better, confirming that preserving full spectral resolution matters.

### Inference Speed

| Call Duration | GPU (3090 Ti) | CPU |
|---------------|--------------|-----|
| 1s | 5.8ms | 12ms |
| 5s | 5.8ms | 25ms |
| 10s | 5.8ms | 49ms |
| 30s | 6.9ms | 241ms |

200x real-time on CPU. Fast enough for any deployment scenario.

## Quick Start

```bash
# Precompute 170-dim features for LibriSpeech IMBE pairs
python -m src.precompute --pairs-dir data/pairs --workers 12

# Train (proof-of-concept 9.2M model)
python -m src.train \
    --pairs-dir data/pairs \
    --librispeech-dir data/LibriSpeech/train-clean-100 \
    --epochs 50 --batch-size 32 --lr 3e-4

# Run inference on a TAP file
python -m src.inference \
    --checkpoint checkpoints/best.pth \
    --tap-file path/to/call.tap

# Streaming demo
python -m src.inference \
    --checkpoint checkpoints/best.pth \
    --npz path/to/file.npz \
    --stream --chunk-ms 500

# Evaluate with beam search + language model
python -m src.eval checkpoints/best.pth \
    --beam --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt
```

## Project Structure

```
imbe_asr/
├── CLAUDE.md              <- Claude Code project instructions
├── README.md              <- You are here
├── .gitignore
├── src/                   <- All model/training/inference code
│   ├── model.py           <- Conformer-CTC architecture (9.2M params)
│   ├── tokenizer.py       <- Character-level CTC tokenizer (40 classes)
│   ├── dataset.py         <- LibriSpeech IMBE dataset + collate_fn + speaker split
│   ├── dataset_p25.py     <- Real P25 TAP dataset (talkgroup-based split)
│   ├── train.py           <- Training loop (CTC loss, cosine LR, W&B)
│   ├── eval.py            <- WER/CER evaluation (standalone + library)
│   ├── decode.py          <- Beam search with KenLM language model
│   ├── inference.py       <- Single-file, TAP file, and streaming inference
│   └── precompute.py      <- frame_vectors -> 170-dim raw_params via libimbe
├── scripts/               <- Data preparation and utility scripts
│   ├── prepare_librispeech.sh  <- Download LibriSpeech + IMBE-encode + precompute
│   ├── prepare_p25_tap.py      <- TAP files -> transcribed NPZ via ASR server
│   └── build_lm.py             <- Build KenLM n-gram language model
├── configs/               <- Training configurations
│   ├── base_9m.yaml       <- POC: d_model=256, 6 layers, 9.2M params
│   └── large_30m.yaml     <- Scale-up: d_model=512, 8 layers, ~30M params
├── data/                  <- Training data (not checked in)
│   ├── pairs/             <- LibriSpeech IMBE NPZ files
│   ├── p25_raw/           <- Real P25 transcribed NPZ files
│   └── lm/               <- Language model artifacts
└── checkpoints/           <- Model checkpoints (not checked in)
```

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- numpy
- `libimbe.so` (IMBE vocoder C library, built from `vocoder/` in parent repo)
- pyctcdecode + kenlm (optional, for beam search)
- wandb (optional, for experiment tracking)
- requests (for P25 TAP data preparation)

## Data Pipeline

### LibriSpeech (pretraining)

```
LibriSpeech FLAC -> IMBE encode (generate_pairs.py) -> NPZ with frame_vectors
    -> precompute (src.precompute) -> NPZ with raw_params (170-dim)
    -> train (src.train) with LibriSpeech transcripts
```

### Real P25 (fine-tuning)

```
P25 TAP JSON (from trunk-recorder IMBE tap socket)
    -> decode through libimbe (raw_params + synthesized WAV)
    -> transcribe WAV via ASR server (Qwen3-ASR)
    -> save NPZ with raw_params + transcript
    -> fine-tune (src.train with --checkpoint)
```

## Next Steps

- **Scale model** to ~30-80M params (d_model=512, 8-12 layers) for broader vocabulary coverage
- **Mixed training** on LibriSpeech + real P25 data for domain coverage without catastrophic forgetting
- **Domain-specific LM** trained on P25 transcripts (unit numbers, 10-codes, street names, medical terminology, general conversation)
- **Larger training set** -- more LibriSpeech splits, real P25 from more agencies/regions
