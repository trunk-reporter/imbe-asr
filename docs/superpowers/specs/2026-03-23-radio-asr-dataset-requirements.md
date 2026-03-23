# Radio ASR Dataset Requirements

## Purpose

This document defines what a training dataset needs to look like for vocoder-parameter ASR models -- models that read raw codec parameters (IMBE, AMBE+2, Codec2, MELPe) directly, without audio reconstruction. Written from the model consumer's perspective, intended as a requirements spec for a separate data collection platform.

This is not a platform design. It does not specify how data is collected, reviewed, or gamified. It specifies what the model training pipeline needs to receive.

## Scope

Multi-codec from the start. The dataset should support:

| Codec | Bitrate | Systems | Frame Rate |
|-------|---------|---------|------------|
| IMBE | 4.4 kbps | P25 Phase 1 | 20ms (50fps) |
| AMBE+2 | 2.4-9.6 kbps | P25 Phase 2, DMR, dPMR, NXDN, D-STAR, Fusion | 20ms (50fps) |
| Codec2 | 0.7-3.2 kbps | FreeDV | 10-40ms (varies) |
| MELPe | 1.2-2.4 kbps | NATO STANAG 4591 | 22.5ms |

New codecs should be addable without changing the dataset format.

## Per-Sample Record

Each sample is one transmission (call). Required fields:

### Core (required)

| Field | Type | Description |
|-------|------|-------------|
| `codec_type` | string | Codec identifier: `"imbe"`, `"ambe2"`, `"codec2"`, `"melpe"` |
| `codec_params` | uint32 array | Raw codewords as received, shape `(N, K)` where K is codec-specific (IMBE: K=8, AMBE+2: varies) |
| `transcript` | string | Uppercased English text |
| `sample_id` | string | Globally unique identifier |

### Codec Metadata (required)

| Field | Type | Description |
|-------|------|-------------|
| `codec_bitrate` | int | Nominal bitrate in bps (e.g., 4400 for IMBE) |
| `codec_mode` | string | Codec sub-mode if applicable (e.g., AMBE+2 rate) |
| `frame_duration_ms` | float | Duration of one frame in milliseconds |
| `n_frames` | int | Number of frames in this sample |
| `duration_s` | float | Total duration in seconds |

### Provenance (required for real captures, optional for synthetic)

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | `"live_capture"`, `"archive"`, `"synthetic"` (re-encoded from audio) |
| `system_type` | string | `"p25"`, `"dmr"`, `"nxdn"`, `"dstar"`, etc. |
| `talkgroup` | int | Talkgroup ID (0 if unknown) |
| `region` | string | Geographic region, coarse (e.g., `"US-OH"`, `"US-CA"`) -- no specific agency names |
| `capture_date` | string | ISO 8601 date (day precision, e.g., `"2026-03-23"`) |

### Quality Metadata (required)

| Field | Type | Description |
|-------|------|-------------|
| `label_method` | string | How the transcript was produced: `"human"`, `"whisper"`, `"qwen3"`, `"consensus"`, `"human_verified"` |
| `label_confidence` | float | 0.0-1.0, producer's confidence in the transcript |
| `verified` | bool | Whether a human has reviewed and accepted this transcript |
| `fec_error_rate` | float | FEC error rate from the radio link (0.0 = clean, higher = degraded) |

### Optional Metadata

| Field | Type | Description |
|-------|------|-------------|
| `src_id` | int | Source radio unit ID |
| `frequency` | int | RF frequency in Hz |
| `encrypted` | bool | Whether the transmission was encrypted (should always be false in dataset -- encrypted calls should be excluded) |
| `tags` | string list | Free-form tags: `["fire", "dispatch", "ems", "traffic_stop", "conversational"]` |
| `notes` | string | Reviewer notes |

## What NOT to Include

- **No audio.** The dataset is codec parameters only. Audio can be reconstructed from the codewords if needed, but storing it doubles the size for no training benefit.
- **No agency names or officer names.** Region codes only. Privacy matters.
- **No encrypted transmissions.** Obvious.
- **No PII in transcripts.** Phone numbers, SSNs, addresses with apartment numbers, names of victims/suspects should be redacted with `[REDACTED]`. Street addresses and unit numbers in dispatch context are fine.

## Storage Format

### Individual samples: Parquet (preferred) or NPZ

For a public HF dataset, Parquet is standard -- it's columnar, compressed, streamable, and HF Datasets loads it natively. Each row is one transmission.

- `codec_params` stored as a binary blob (the raw uint32 array, serialized)
- All metadata as typed columns
- Partitioned by `codec_type` for efficient filtering

For local training, the existing NPZ format or memory-mapped binary (`MmapIMBEDataset`) can be generated from the Parquet source.

### Dataset splits

- No pre-defined train/val/test splits in the dataset itself
- Include enough metadata (talkgroup, region, capture_date) for consumers to create their own splits
- Recommended split strategy: by region (geographic generalization) or by talkgroup (speaker generalization)

## Pre-extracted Features (Optional Companion)

The raw codewords require a codec library (libimbe, libambe, etc.) to decode into the feature vectors the model consumes. For convenience, the dataset may optionally include pre-extracted features:

| Field | Type | Description |
|-------|------|-------------|
| `features` | float32 array | Decoded vocoder parameters, shape `(N, D)` where D is codec-specific (IMBE: D=170) |
| `feature_version` | string | Version of the extraction code (for reproducibility) |

This is optional because:
1. Codec libraries may improve over time (better libimbe = better features)
2. Different consumers may want different feature representations
3. The raw codewords are the ground truth

If included, features should be stored in a separate Parquet file or partition, linked by `sample_id`.

## Volume Targets

For meaningful model improvement, rough targets by priority:

| Tier | Hours | Impact |
|------|-------|--------|
| Minimum viable | 100h verified | Enough for fine-tuning, significant P25 WER improvement |
| Good | 500h verified | Robust domain adaptation, multiple regions |
| Excellent | 2000h+ verified | Train-from-scratch quality, multi-codec |

Current state: ~20h pseudo-labeled (unverified) IMBE. The 1220h of IMBE-encoded LibriSpeech/TEDLIUM/GigaSpeech is synthetic (software-encoded, not real radio).

## Diversity Requirements

A useful dataset must cover:

- **Multiple regions** -- at least 5-10 US states/metro areas to avoid geographic vocabulary bias
- **Multiple agency types** -- law enforcement, fire, EMS, public works, transit (P25 is not just police dispatch)
- **Multiple codecs** -- IMBE at minimum, AMBE+2 for DMR coverage
- **Signal quality range** -- clean transmissions AND degraded/noisy ones (the model needs to handle real-world conditions)
- **Conversation variety** -- dispatch, field reports, casual conversation, medical, traffic, fire ground operations

## Licensing

For a public dataset:

- Contributor license: contributors must affirm they have the right to share the captured radio transmissions (P25/public safety radio is generally unencrypted and legally receivable in the US, but this varies by jurisdiction)
- Dataset license: CC-BY-SA 4.0 or similar -- open for research and commercial use with attribution
- No warranty on transcript accuracy -- this is crowdsourced data

## Integration with IMBE-ASR Training Pipeline

The current training pipeline expects:

1. NPZ files with `raw_params` (float32, shape `(N, 170)`) and `transcript` (string)
2. Or memory-mapped binary format via `MmapIMBEDataset`

A conversion script (`dataset_to_npz.py` or similar) should translate from the Parquet dataset format to the training format:

```
parquet (codec_params uint32) -> codec library decode -> raw_params float32 -> NPZ or mmap
```

This conversion is the consumer's responsibility, not the dataset's. The dataset stores the canonical representation (raw codewords).

## Multi-Codec Model Architecture Implications

For a model that handles multiple codecs, the dataset needs to cleanly separate:

1. **Codec-specific data** -- the raw codewords and their native dimensionality
2. **Codec-agnostic data** -- the transcript, metadata, and quality signals

The model architecture uses codec-specific input projections (`Linear(D_codec, d_model)`) mapping each codec's native parameter space into a shared encoder. The dataset format supports this by storing native codewords per codec type, not a pre-normalized fixed-size representation.

Training a multi-codec model:
- Filter by `codec_type` to get codec-specific batches (or mix with a codec identifier token)
- Each codec's features are extracted independently with the appropriate library
- The shared conformer encoder learns across all codecs

## Summary

The dataset is a collection of (codewords, transcript) pairs with rich metadata. Store raw codec parameters (not decoded features), use Parquet for distribution, include enough provenance and quality metadata for consumers to filter and split as needed. Privacy-conscious, multi-codec from day one, designed to scale from 100h to 2000h+.
