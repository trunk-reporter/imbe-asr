# CLAUDE.md

## Project Purpose

Speech recognition directly from P25 IMBE vocoder parameters -- skip lossy audio reconstruction entirely and go straight from decoded codec parameters to text. This is a standalone project for the ASR model, training pipeline, and inference tooling.

The parent project (`~/p25_audio_quality/`) contains trunk-recorder, the IMBE tap, vocoder library, and training data generation. This repo contains only the clean ASR code.

## Key Concepts

**P25 radio is general conversational English.** It is NOT limited to dispatch codes or 10-codes. P25 covers law enforcement, fire, EMS, public works, transit, schools, bus drivers, ODOT workers -- full breadth of everyday English conversation. The ASR model must handle this vocabulary breadth.

**170-dim raw IMBE parameters** are the canonical input representation. Each 20ms frame is decoded from 8 codeword parameters `u[0..7]` through libimbe into:
- `[0]` f0 (fundamental frequency)
- `[1]` L (number of harmonics, 0-56)
- `[2:58]` spectral amplitudes (56 slots, zero-padded)
- `[58:114]` voiced/unvoiced flags per harmonic (56 slots)
- `[114:170]` binary harmonic validity mask (1=real, 0=pad)

The mask is critical -- without it the model can't distinguish zero-energy harmonics from padding. This "middle path" decisively beat both raw 8-dim codewords and 20-dim BFCCs (40.6% vs 56% greedy WER).

## Repository Layout

```
src/              All model/training/inference code (run as python -m src.*)
  model.py        Conformer-CTC architecture
  tokenizer.py    Character-level CTC tokenizer (40 classes)
  dataset.py      LibriSpeech IMBE dataset + collate_fn + speaker split
  dataset_p25.py  Real P25 TAP dataset (splits by talkgroup)
  train.py        Training loop (CTC, cosine LR, W&B)
  eval.py         WER/CER evaluation
  decode.py       Beam search with KenLM
  inference.py    Single-file, TAP file, and streaming inference
  precompute.py   frame_vectors -> 170-dim raw_params via libimbe

scripts/          Data preparation (not part of src package)
  prepare_librispeech.sh   Download + IMBE-encode LibriSpeech
  prepare_p25_tap.py       TAP files -> transcribed NPZ via Qwen3-ASR
  build_lm.py              Build KenLM n-gram LM

configs/          Training configurations (base_9m.yaml, large_30m.yaml)
data/             Training data (not checked in)
checkpoints/      Model checkpoints (not checked in)
```

## How to Run

All `src/` modules run from the project root:

```bash
# Precompute features
python -m src.precompute --pairs-dir data/pairs --workers 12

# Train
python -m src.train --pairs-dir data/pairs \
    --librispeech-dir data/LibriSpeech/train-clean-100

# Inference on a TAP file
python -m src.inference --checkpoint checkpoints/best.pth \
    --tap-file path/to/call.tap

# Streaming demo
python -m src.inference --checkpoint checkpoints/best.pth \
    --npz path/to/file.npz --stream --chunk-ms 500

# Evaluate with beam search + LM
python -m src.eval checkpoints/best.pth --beam \
    --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt
```

## Dependencies

- Python 3.10+, PyTorch 2.0+, numpy
- `libimbe.so` -- IMBE vocoder C library (built from `~/p25_audio_quality/vocoder/`)
- pyctcdecode + kenlm (optional, for beam search)
- wandb (optional, for experiment tracking)
- requests (for prepare_p25_tap.py, talks to Qwen3-ASR server)

## External Resources

- **libimbe.so** locations: `./vocoder/libimbe.so` or `/mnt/disk/p25_train/vocoder/libimbe.so`
- **Training data on sarah** (10.2.2.148): `/mnt/disk/p25_train/`
- **Qwen3-ASR server**: `http://localhost:8765/v1/audio/transcriptions` (P25-tuned, on the local A5000)
- **W&B project**: `imbe-asr` under `luxprimatech`

## Architecture Details

**Conformer-CTC** (proof-of-concept 9.2M params):
- Input projection: Linear(170, 256) + LayerNorm + Dropout
- Encoder: 6x ConformerBlock (FFN/2 -> MHSA -> ConvModule -> FFN/2 -> LN)
- CTC head: Linear(256, 40) -> log_softmax
- ConvModule: pointwise(GLU) -> depthwise(k=31) -> BN -> Swish -> pointwise
- Sinusoidal positional encoding, padding mask from input_lengths

**Tokenizer**: blank(0) + A-Z(1-26) + 0-9(27-36) + space(37) + apostrophe(38) = 40 classes

**Normalization**: per-dimension z-score computed from training set, saved as `stats.npz` alongside checkpoints. Pretrained stats must be reused for fine-tuning and inference.

## Current Results

- LibriSpeech-IMBE greedy: 40.6% WER, 14.6% CER (epoch 36)
- With beam search + 5-gram LM: ~28-30% WER
- Inference: 5.8ms GPU / 49ms CPU for 10s call (200x real-time on CPU)
- Real P25 inference: recognizable fragments but significant domain gap from LibriSpeech-only training

## Conventions

- Use `%` string formatting (not f-strings) in print statements for consistency with the codebase
- All text is uppercased before tokenization
- NPZ files use `raw_params` key for 170-dim features, `transcript` for text
- Speaker-based splits for LibriSpeech, talkgroup-based splits for P25
- Checkpoints save full config dict so models are self-describing
