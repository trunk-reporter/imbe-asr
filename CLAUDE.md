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
  model.py        Conformer-CTC architecture (scalable: 9M to 290M params)
  tokenizer.py    Character-level CTC tokenizer (40 classes)
  dataset.py      LibriSpeech IMBE dataset + collate_fn + speaker split
  dataset_unified.py  Multi-source dataset (LibriSpeech/TEDLIUM/GigaSpeech) + MmapIMBEDataset
  dataset_p25.py  Real P25 TAP dataset (splits by talkgroup)
  train.py        Training loop (CTC, cosine LR, W&B, DDP multi-GPU)
  eval.py         WER/CER evaluation
  decode.py       Beam search with KenLM
  inference.py    Single-file, TAP file, --watch mode, and streaming inference
  live.py         Live P25 transcription from trunk-recorder IMBE tap socket
  precompute.py   frame_vectors -> 170-dim raw_params via libimbe

scripts/          Data preparation and deployment (not part of src package)
  prepare_librispeech.sh     Download + IMBE-encode LibriSpeech train-clean-100
  prepare_librispeech_full.sh  IMBE-encode LibriSpeech 360+500 splits
  prepare_tedlium.py         TEDLIUM 3 -> manifest TSV -> IMBE-encoded NPZ
  prepare_gigaspeech.py      GigaSpeech S -> manifest TSV -> IMBE-encoded NPZ
  imbe_encode.py             Generic audio-to-IMBE encoder (any dataset via TSV manifest)
  prepare_p25_tap.py         TAP files -> transcribed NPZ via Qwen3-ASR
  pseudo_label.py            Multi-ASR pseudo-labeling (Whisper + Qwen3) for P25 data
  finetune_p25.py            P25 fine-tuning with mixed base+P25 data
  whisper_server.py          Local Whisper large-v3 transcription server (port 8766)
  pack_dataset.py            Pack NPZ files into memory-mapped binary (parallel scanning)
  sweep_agent.py             W&B hyperparameter sweep agent
  build_lm.py                Build KenLM n-gram LM
  vast_push.sh               Push code/data to vast.ai instance
  vast_launch.sh             Launch training on vast.ai GPU instance
  vast_sweep.sh              Run hyperparameter sweep on vast.ai

configs/          Training configurations
  base_9m.yaml       Original 9.2M proof-of-concept
  large_30m.yaml     30M param config
  data_expanded.yaml ~1220h multi-source data config (LS 100+360+500, TEDLIUM, GigaSpeech)
  data_eddie.yaml    Data paths for local GPU workstation
  data_sarah.yaml    Data paths for training server
  sweep.yaml         W&B Bayesian hyperparameter sweep config

data/             Training data (not checked in)
checkpoints/      Model checkpoints (not checked in)
```

## How to Run

All `src/` modules run from the project root:

```bash
# Precompute features (LibriSpeech or any dataset)
python -m src.precompute --pairs-dir data/pairs --workers 12

# Train (single-source, backward compatible)
python -m src.train --pairs-dir data/pairs \
    --librispeech-dir data/LibriSpeech/train-clean-100 \
    --epochs 50 --batch-size 32 --lr 3e-4

# Train (multi-source expanded dataset)
python -m src.train --data-config configs/data_expanded.yaml \
    --d-model 512 --n-layers 8 --n-heads 8 --d-ff 2048 \
    --epochs 30 --batch-size 32 --lr 2e-4

# Multi-GPU training (DDP via torchrun)
torchrun --nproc_per_node=2 -m src.train \
    --data-config configs/data_expanded.yaml \
    --epochs 30 --batch-size 32 --lr 2e-4

# P25 fine-tuning (mixed base + pseudo-labeled P25 data)
python3 scripts/finetune_p25.py \
    --checkpoint checkpoints/eddie_512d/best.pth \
    --p25-dir data/p25_labeled \
    --base-mmap data/packed_eddie \
    --output checkpoints/p25_finetuned --epochs 10 --lr 5e-5

# Inference on a TAP file (binary or JSON format)
python -m src.inference --checkpoint checkpoints/p25_finetuned/best.pth \
    --tap-file path/to/call.tap

# Watch directory for new .tap files (live transcription)
python -m src.inference --checkpoint checkpoints/p25_finetuned/best.pth \
    --watch ~/trunk-recorder/audio/

# Streaming demo
python -m src.inference --checkpoint checkpoints/best.pth \
    --npz path/to/file.npz --stream --chunk-ms 500

# Live socket-based transcription (trunk-recorder IMBE tap)
python -m src.live --checkpoint checkpoints/p25_finetuned/best.pth \
    --socket /tmp/imbe_tap.sock

# Evaluate with beam search + LM
python -m src.eval checkpoints/best.pth --beam \
    --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt

# Pseudo-label P25 .tap files (requires Whisper + Qwen3 servers)
python3 scripts/pseudo_label.py \
    --tap-dir ~/trunk-recorder/audio/ \
    --output-dir data/p25_labeled
```

## Dependencies

- Python 3.10+, PyTorch 2.0+, numpy, pyyaml
- `libimbe.so` -- IMBE vocoder C library (built from `~/p25_audio_quality/vocoder/`)
- pyctcdecode + kenlm (for beam search decoding)
- wandb (for experiment tracking and hyperparameter sweeps)
- soundfile (for audio I/O in pseudo-labeling pipeline)
- requests (for pseudo_label.py and prepare_p25_tap.py, talks to ASR servers)
- awscli (for S3 checkpoint staging to vast.ai via OVH)

## External Resources

- **libimbe.so** locations: `./vocoder/libimbe.so` or `/mnt/disk/p25_train/vocoder/libimbe.so`
- **Qwen3-ASR server**: `http://localhost:8765/v1/audio/transcriptions` (for pseudo-labeling)
- **W&B project**: `imbe-asr` under `luxprimatech`

## Architecture Details

**Conformer-CTC** (scalable architecture, currently trained at two sizes):

| Config | d_model | layers | heads | d_ff | params |
|--------|---------|--------|-------|------|--------|
| Base (current best) | 512 | 8 | 8 | 2048 | 48.6M |
| Large (1 epoch on eddie) | 1024 | 12 | 16 | 4096 | 290M |

Hyperparameter sweep (Bayesian, W&B) showed optimal architecture is d=1024, 12 layers, ff_mult=4. Depth (12 layers) matters more than width.

- Input projection: Linear(170, d_model) + LayerNorm + Dropout
- Optional 2x strided conv subsampling (50fps -> 25fps)
- Encoder: N x ConformerBlock (FFN/2 -> MHSA -> ConvModule -> FFN/2 -> LN)
- CTC head: Linear(d_model, 40) -> log_softmax
- ConvModule: pointwise(GLU) -> depthwise(k=31) -> BN -> Swish -> pointwise
- MHSA: fused Q/K/V projection, F.scaled_dot_product_attention (FlashAttention-2 on PyTorch 2.0+)
- Sinusoidal positional encoding, padding mask from input_lengths

**IMBE encoder delay compensation**: Software-encoded training data (LibriSpeech, TEDLIUM, GigaSpeech) has a 2-frame analysis delay -- encoded frame N describes audio at frame N-2. All dataset classes trim the first `ENCODER_DELAY_FRAMES = 2` frames. Real P25 TAP data (captured from radios) is NOT affected.

**P25 silence frame stripping**: P25 interleaves voice IMBE frames with signaling (LICH, HDU, LDU headers) that decode to zero spectral energy. These are detected by decoding through libimbe and checking for zero energy in spectral amplitude bands [2:58], then removed before inference.

**Tokenizer**: blank(0) + A-Z(1-26) + 0-9(27-36) + space(37) + apostrophe(38) = 40 classes

**Normalization**: per-dimension z-score computed from training set, saved as `stats.npz` alongside checkpoints. Pretrained stats must be reused for fine-tuning and inference.

**Training data**: ~1220 hours across LibriSpeech (100+360+500h), TEDLIUM 3 (~452h), GigaSpeech S (~250h), plus ~20 hours of P25 pseudo-labeled data. All audio is IMBE-encoded through libimbe to produce 170-dim frame parameters. Multi-source config in `configs/data_expanded.yaml`. Memory-mapped binary format (`MmapIMBEDataset`) for fast loading on remote GPU instances.

## Current Results

**Base model (48.6M, d=512/8L/8H, expanded dataset)**:
- LibriSpeech-IMBE greedy: 18.9% WER, 6.8% CER (epoch 25)
- With beam search + 5-gram KenLM: significant further improvement

**P25-finetuned (48.6M, epoch 9, mixed base+P25 data)**:
- LibriSpeech-IMBE: 19.2% WER (slight regression from base)
- Real P25: substantially better -- produces readable fire dispatch transcriptions (e.g., "BATTALION 60 ENGINE 62 MEDIC 61...")
- Live --watch mode tested and working on real trunk-recorder .tap files

**Inference speed**: 5.8ms GPU / 49ms CPU for 10s call (200x real-time on CPU)

**Previous proof-of-concept (9.2M, d=256/6L)**: 40.6% WER greedy, ~28-30% with LM

## Conventions

- Use `%` string formatting (not f-strings) in print statements for consistency with the codebase
- All text is uppercased before tokenization
- NPZ files use `raw_params` key for 170-dim features, `transcript` for text
- Speaker-based splits for LibriSpeech, talkgroup-based splits for P25
- Checkpoints save full config dict so models are self-describing
