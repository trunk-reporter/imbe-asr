# Radio ASR Dataset Platform -- Handoff from IMBE-ASR

## What This Is

Context and requirements for building a crowdsourced radio ASR dataset platform. Written by the IMBE-ASR model training side to give the platform project everything it needs to get started.

## The Problem We're Solving

We have a speech recognition model that reads raw vocoder parameters from digital radio (P25, DMR, etc.) and produces text -- no audio reconstruction needed. It works: 1.9% WER on LibriSpeech-IMBE, runs on a Raspberry Pi.

But it was trained on synthetic data -- clean speech (LibriSpeech, TEDLIUM, GigaSpeech) software-encoded through the IMBE vocoder. Real radio has FEC errors, channel noise, cross-talk, background noise, and radio-specific vocabulary that synthetic data can't capture. We have ~20 hours of real P25 pseudo-labeled data, which is not enough to meaningfully improve a 290M parameter model.

We need hundreds to thousands of hours of real radio transcription data, verified by humans.

## The Data Pipeline (High Level)

```
Contributors run trunk-recorder     Centralized server         Human reviewers
with symbolstream plugin       -->  receives codec symbols --> verify/correct
                                    auto-labels via            transcripts
                                    Whisper + Qwen3 ASR   -->
                                                               Verified data
                                                               published to HF
```

### Step 1: Capture (Distributed)

Contributors run trunk-recorder (our fork: https://github.com/trunk-reporter/trunk-recorder) with two plugins:

- **symbolstream** (https://github.com/trunk-reporter/symbolstream) -- streams raw IMBE/AMBE codec symbols over TCP. These are the 8 codeword parameters per 20ms frame, before any audio reconstruction.
- **audio_rtc** (https://github.com/MimoCAD/tr-plugin-rtc) -- streams real-time PCM audio over TCP. Same transmissions, decoded to audio. Needed for human review playback and for running audio-based ASR for pseudo-labeling.

Both plugins connect outward to a configured server address. The symbolstream plugin config looks like:
```json
{
    "name": "symbolstream",
    "library": "libsymbolstream_plugin",
    "streams": [{
        "address": "data-server.example.com",
        "port": 9090,
        "TGID": 0,
        "useTCP": true,
        "sendJSON": true
    }]
}
```

Wire format is documented in the symbolstream README and in our spec at `docs/superpowers/specs/2026-03-23-symbolstream-client-design.md`.

### Step 2: Decode + Auto-Label (Centralized)

The server receives raw codec symbols and audio simultaneously. It needs to:

1. **Store raw codec symbols** per call (the canonical training data)
2. **Reconstruct audio** from codec symbols via libimbe (for reviewer playback)
3. **Run auto-labeling** on the audio:
   - Whisper large-v3 (general ASR)
   - Qwen3-ASR P25 fine-tune (https://huggingface.co/trunk-reporter/qwen3-asr-p25-0.6B, tuned for P25)
   - When both agree (low edit distance): high confidence auto-label
   - When they disagree: flag for human review
4. **Store metadata** from the stream: talkgroup, source ID, timestamp, FEC error count, system name

We already have a working pseudo-labeling pipeline (`scripts/pseudo_label.py` in the imbe-asr repo) that does the Whisper + Qwen3 consensus. It works on .tap files but the same logic applies to streamed data.

Qwen3-ASR server: https://github.com/trunk-reporter/qwen3-asr-server (OpenAI-compatible API)

### Step 3: Human Review (Crowdsourced)

Reviewers see:
- Audio playback (reconstructed from codec symbols)
- Auto-generated transcript
- Talkgroup name and category for context
- FEC error rate (signal quality indicator)

They can:
- Accept the transcript as-is
- Edit and correct it
- Flag it as garbage (unintelligible, music, tones, encrypted bleed-through)
- Skip

The user envisions gamification -- leaderboards, team rankings, competitive quality metrics. Platform design handles this.

### Step 4: Publish

Verified data is published to Hugging Face as a public dataset under the trunk-reporter org.

## Dataset Format Spec

Full spec: `docs/superpowers/specs/2026-03-23-radio-asr-dataset-requirements.md` (in the imbe-asr repo)

Key points:

### Per-sample fields

```
codec_type          string    "imbe", "ambe2", "codec2", "melpe"
codec_params        uint32[]  Raw codewords, shape (N, K), K=8 for IMBE
transcript          string    Uppercased English text
sample_id           string    Globally unique
audio               bytes     Reconstructed audio (OGG) for review
talkgroup           int       Talkgroup ID
talkgroup_tag       string    Human-readable name ("Hamilton County Fire")
talkgroup_group     string    Category ("Fire Dispatch", "EMS", etc.)
region              string    "US-OH", "US-CA", etc.
source              string    "live_capture", "archive", "synthetic"
system_type         string    "p25", "dmr", "nxdn", etc.
label_method        string    "whisper", "qwen3", "consensus", "human_verified"
label_confidence    float     0.0 - 1.0
verified            bool      Human reviewed and accepted
fec_error_rate      float     Signal quality (0.0 = clean)
capture_date        string    ISO 8601 date
duration_s          float     Seconds
n_frames            int       Frame count
```

### Storage

Parquet for HF distribution. Partitioned by codec_type. Audio as binary blob (OGG compressed).

### Privacy

- No PII in transcripts (redact with `[REDACTED]`)
- Talkgroup names are public info (RadioReference) -- ok to include
- No officer names or badge numbers in metadata
- No encrypted transmissions
- Region codes instead of specific agency identifiers in top-level metadata

## Codec Libraries

The platform needs these for decoding codec symbols to audio (for review) and to feature vectors (for optional pre-extraction):

- **libimbe** -- IMBE vocoder. Open-source, reverse-engineered. We build from source at https://github.com/trunk-reporter/trunk-recorder (vocoder/ directory).
- **libambe** -- AMBE+2 vocoder. More restricted. md380-emu or DSD can decode. May need a hardware dongle (ThumbDV/DV3000) for some modes.
- **Codec2** -- Fully open source. https://github.com/drowe67/codec2
- **MELPe** -- Reference implementations exist but may have licensing constraints.

Start with IMBE (P25 Phase 1). It's the most common, we have the tooling, and it's what our current model is trained on.

## Volume Targets

| Tier | Hours | What it enables |
|------|-------|-----------------|
| 100h verified | Fine-tuning, real WER improvement on P25 |
| 500h verified | Robust multi-region P25, start AMBE+2 |
| 2000h+ verified | Train-from-scratch, multi-codec |

Current: ~20h pseudo-labeled (unverified) IMBE from one region (Ohio).

## What the Model Training Side Consumes

The IMBE-ASR training pipeline (`src/train.py`) expects:

1. NPZ files: `raw_params` (float32, `(N, 170)`) + `transcript` (string)
2. Or memory-mapped binary via `MmapIMBEDataset`

A conversion from the Parquet dataset to training format:
```
parquet -> extract codec_params -> decode via libimbe -> 170-dim float32 -> NPZ/mmap
```

We'll write this converter. The platform just needs to produce valid Parquet per the spec.

## Existing Trunk-Reporter Infrastructure

- **GitHub org**: https://github.com/trunk-reporter
- **HF org**: https://huggingface.co/trunk-reporter
- **trunk-recorder fork**: https://github.com/trunk-reporter/trunk-recorder (voice_codec_data callback)
- **symbolstream plugin**: https://github.com/trunk-reporter/symbolstream
- **tr-engine**: https://github.com/trunk-reporter/tr-engine (backend API for trunk-recorder MQTT messages, REST + SSE)
- **tr-dashboard**: https://github.com/trunk-reporter/tr-dashboard
- **Qwen3-ASR server**: https://github.com/trunk-reporter/qwen3-asr-server

## Improvements Noted for Symbolstream

From reviewing MimoCAD/tr-plugin-rtc (audio_rtc), symbolstream should add:
- `src_tag` / `talkgroup_tag` from TR's lookup tables
- Talkgroup patching via `get_talkgroup_patch()`
- Source ID fallback from last transmission
- Granular `sendCallStart` / `sendCallEnd` config flags

These would enrich the metadata available to the platform without extra lookups.

## Questions for the Platform Project

1. **Hosting** -- where does the centralized ingest server run? Needs GPU for Whisper/Qwen3 auto-labeling.
2. **Review UI** -- web app? Mobile-friendly? What framework?
3. **Contributor auth** -- how do contributors register and connect their trunk-recorder instances?
4. **Scale** -- how many simultaneous contributor streams can the server handle?
5. **Moderation** -- who reviews the reviewers? Quality control on the human verification.

## Contact

This document was written from the IMBE-ASR model training perspective. The dataset spec and all referenced code are in the imbe-asr repo:

- **Code**: https://github.com/trunk-reporter/imbe-asr
- **Models**: https://huggingface.co/collections/trunk-reporter/imbe-asr-speech-recognition-from-vocoder-parameters-69c0a4f68ef670b5bf68449d
- **Dataset spec**: `docs/superpowers/specs/2026-03-23-radio-asr-dataset-requirements.md`
