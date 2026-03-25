# IMBENet Audio Reconstruction — Artifact Root Cause Analysis

**Date:** 2026-03-24
**Author:** Research agent (claude-sonnet-4-6)
**Sources:** `~/p25_audio_quality/train/imbenet/`, `~/p25_vocoder/docs/project_status.md`, all training logs, wandb run files, diagnostic scripts

---

## Executive Summary

"IMBENet artifacts" covers two distinct failure modes across two architectures:

1. **IMBENet (GRU autoregressive)** — training failure, not an audio quality issue. Loss stuck at 0.119 from epoch 1. The model never learns. Outputs are spectral-mean noise/buzz. This is caused by fundamental misalignment between autoregressive waveform synthesis and STFT loss.

2. **Vocos-IMBE (ISTFTHead non-autoregressive)** — trains successfully but produces a **50Hz comb filter / flanging artifact** in output audio. Root cause: IMBE's non-overlapping 20ms frames create step-function feature changes at frame boundaries; the ISTFTHead predicts phase independently per STFT frame with zero temporal context; adjacent ISTFT windows with inconsistent phases cause periodic constructive/destructive interference.

GAN training fixed the core buzz (one epoch of fine-tuning did more than 15 epochs of reconstruction loss), but the 50Hz comb filter was only partially addressed (2x upsample reduced it; was not fully eliminated). A secondary issue on real P25 data is encoder distribution mismatch.

---

## 1. IMBENet (GRU) — Training Failure Analysis

### 1.1 Evidence

All training runs show loss plateau from the **first 100 batches**:

| Run | Config | Epoch 1 batch-100 loss | Best epoch loss |
|-----|--------|------------------------|-----------------|
| vrfpkhom (ep 3+) | lr=5e-4, bs=512, seq=15 | 0.1224 | ~0.1207 (epoch 4) |
| p9jpqs89 (ep 3+) | lr=5e-4, bs=512, seq=15 | 0.1217 | ~0.1207 (epoch 5) |
| ijl9y6du (ep 7+) | lr=1e-3, bs=512, fresh optim | 0.1187 | 0.1188 (flat) |

The loss shows essentially **zero improvement** across 7+ epochs (700+ hours of GPU compute on a 3090 Ti). For comparison, the Vocos-IMBE loss improves from 55 → 12 in 8 epochs on the same dataset.

The inference script (`inference.py`) confirms outputs are near-uniform buzz: correlation between predicted and ground truth audio is reported alongside MSE. A correlation near zero with finite MSE confirms the model has learned to output constant or stationary noise.

### 1.2 Root Cause: STFT Loss Cannot Drive Autoregressive Phase Learning

The IMBENet training loop uses multi-resolution STFT loss — L1 on sqrt-compressed magnitudes at 6 resolutions (n_fft=80 to 2560). This loss works by:

1. Computing STFT of predicted and ground truth audio
2. Comparing sqrt(|STFT|) per time-frequency bin

For a GRU synthesizing 40 samples per subframe, there is **no mechanism to control phase**. The GRU receives pitch excitation (a lookup from previous output) and conditioning (spectral envelope), but the gradient from STFT loss must flow backward through:

```
STFT loss → sqrt(|STFT|) → STFT → 40 samples → GRU output → GRU state → ...
```

The phase of each STFT bin depends on the exact timing of every sample in the window. The GRU has no "phase register" to update — it learns amplitude but cannot learn absolute phase alignment across overlapping STFT windows.

**What the model learns instead:** The minimum of the STFT loss under random phase is achieved by predicting the mean magnitude spectrum. This is equivalent to outputting a stationary noise process whose power spectrum matches the average target spectrum — a buzzy, voiced-sounding drone. The model hits this floor at ~0.119 by epoch 1 and cannot improve because every sample offset that improves phase for one window degrades it for adjacent overlapping windows.

This is confirmed in `p25_vocoder/docs/project_status.md`: *"15 epochs of reconstruction-only training produced zero perceptual improvement after epoch 1. The buzz was the spectral mean — reconstruction loss converges to the average waveform immediately."*

### 1.3 Why GAN Training Also Failed

Six adversarial configurations were tried with the FARGAN discriminator (FDMResDisc):

| Run | Config | Outcome |
|-----|--------|---------|
| 1 | lr=5e-4, pretrained, 1:1 ratio | Collapsed batch 800, wc→0.15 |
| 2 | lr=2e-5, pretrained | Collapsed batch 812, wc→0.118 |
| 3 | lr=2e-6 (paper LR), pretrained | Collapsed epoch 3, wc→0.089 |
| 4a | lr=1e-4, from scratch | Collapsed batch 350, wc→0.148 |
| 4b | lr=1e-5, from scratch | Collapsed batch 1355, wc→0.090 |
| 5 | lr=2e-6, 5:1 disc ratio | Collapsed batch 852, wc→0.114 |

Winning chance (wc) collapsed to 0.08-0.15 in all cases — the discriminator dominated immediately. **Root causes:**

- FDMResDisc was designed for Opus SILK features (2560-dim spectral context), not IMBE's 128-bin interpolated envelope. The discriminator found trivial narrow-bandwidth shortcuts.
- IMBE's binary per-harmonic voicing and coarse spectral quantization create distribution artifacts that the discriminator exploits without learning perceptually meaningful features.
- The GRU generator's sequential structure (40 samples per subframe, 8 subframes per frame) makes gradient signal from discriminator arrive many steps after the synthesis decision — the discriminator sees the full waveform while the generator has no temporal context to use the feedback.

### 1.4 Prior FARGAN-based Artifacts (Historical)

The first version of the GRU vocoder (direct FARGAN port) showed an additional artifact: **25-45% of output frames were exactly zero** during voiced speech. Root cause: the scalar gain normalization mechanism (applied to excitation before injecting into the GRU) produced near-zero gains when IMBE's spectral amplitude distributions saturated the tanh activations.

This was fixed in IMBENet by removing the explicit gain mechanism and switching from tanh to GELU activations in IMBECond. The gain instability no longer appears. The remaining problem is the STFT loss plateau — the gain fix alone was not sufficient.

---

## 2. Vocos-IMBE — Flanging / 50Hz Comb Filter

### 2.1 Evidence

Quantitative measurements from `diagnose_flanging_v2.py` and `flanging_final_diagnosis.md`:

| Diagnostic | Measurement | Interpretation |
|-----------|-------------|----------------|
| Cepstral peak at quefrency=320 samples | **37.6× local neighborhood** | Frame-rate comb filter confirmed |
| ACF at 1 frame (320 samp) vs 1.5 frames (480 samp) | **0.43 vs 0.006** | Extreme frame-boundary periodicity |
| Higher-order cepstral peaks (640, 960, 1280) | Strong harmonics | Multi-order comb, not simple echo |
| Spectral flatness pred/GT | 0.73 | Significant comb filtering |
| Bandwidth test (0-4k vs 4-8k) | Equal error in both bands | Not a bandwidth extension issue |

The artifact manifests as **audible 50Hz flanging** — a metallic, swirling quality that tracks the IMBE frame rate (50 frames/second = 20ms = 320 samples at 16kHz).

### 2.2 Root Cause: Step-Function Features + Per-Frame Phase Prediction

Two factors combine:

**Factor 1: Non-overlapping IMBE frames create abrupt feature discontinuities.**

IMBE is a non-overlapping 20ms frame codec. Adjacent frames share **zero audio content** (unlike mel spectrograms which use overlapping windows and share ~75% of content). The decoded log spectral envelope, voicing, and pitch change as step functions at every frame boundary.

**Factor 2: ISTFTHead predicts phase independently per STFT frame.**

The ISTFTHead applies a pointwise `nn.Linear` to each backbone output position, predicting magnitude and phase for one STFT frame. There is **zero temporal context** in the head itself — consecutive frames' phases are predicted independently, even though the ISTFT overlap-add requires consecutive windows to have consistent phase relationships.

**Interaction:** With n_fft=1280 and hop=320, each audio sample is the sum of 4 overlapping ISTFT windows. At a frame boundary, the conditioning steps abruptly, which causes the backbone to output different features on either side of the boundary. The ISTFTHead's independently-predicted phases then produce inconsistent overlap-add, causing constructive/destructive interference at multiples of 320 samples — a 50Hz comb filter.

### 2.3 Contributing Factor: sa Contamination (Now Fixed)

The original data pipeline used a sequential decoder per utterance. `decode_params()` is stateful — it runs IMBE synthesis internally, updating phase accumulators that feed back into spectral amplitude extraction for subsequent frames.

**Measured impact:** Sequential vs fresh decoder per frame showed 31% std difference in log spectral envelope values (sa diff = 0.71 nats). f0, L, and b0 were unaffected (synthesized internally but not returned). This added 31% extra noise to the step-function discontinuities at frame boundaries, amplifying the flanging.

**Fix applied:** `prepare_imbenet_data.py` now creates a fresh decoder per frame. This eliminated the sa contamination and improved loss convergence (old run plateaued at ~13.5; new run reached 12.3). The flanging persisted — confirming it is fundamentally architectural, not a data pipeline issue.

### 2.4 What Was Tried for Flanging

| Fix | Description | Outcome |
|-----|-------------|---------|
| Fresh decoder per frame | Eliminated sa contamination (31% std) | Better loss, flanging persists |
| UpsampleBridge 2x | hop 320→160, backbone at 50fps, ISTFT at 100fps | ACF 0.43→0.09, perceptually less flanging |
| UpsampleBridge 4x | hop 80, even more overlap | Always diverges or underperforms |
| Phase continuity loss | Penalize deviation from expected phase advance | No perceptual difference, weights 0.0–2.0 |
| Anti-periodicity loss | Penalize ACF at multiples of 320 | No perceptual difference |
| IF Head | Predict cumulative instantaneous frequency instead of absolute phase | Eliminated 50Hz comb (0.587→0.021 cepstral peak), but introduced phase drift/robotic artifact |
| GAN fine-tuning (1 epoch) | Added MPD + MRD discriminators | Fixed spectral mean buzz, "actually listenable now" |
| GAN fine-tuning (multiple epochs) | More GAN epochs | Echo/doubling artifact introduced |

### 2.5 Encoder Distribution Mismatch (Real P25 Data)

When running on real P25 radio captures:

| Source | f0 | sa_sum | Notes |
|--------|----|--------|-------|
| libimbe (software, training data) | 174.5 Hz | 823.7 | Used for all training |
| Real P25 radio (DVSI hardware) | 217.7 Hz | 310.1 | Real trunk-recorder captures |
| Ratio | 0.80× | 2.66× | Large mismatch in spectral energy |

The model trained on libimbe features will produce systematically different output when conditioned on real P25 features — louder/brighter than expected given the 2.66× sa_sum difference. This manifests as spectral energy distribution artifacts on real P25 audio, distinct from the flanging issue.

---

## 3. Current State Summary

| Architecture | Status | Primary Artifact | Root Cause |
|-------------|--------|-----------------|------------|
| IMBENet (GRU) | **Training failed** | Spectral-mean buzz, silence | STFT loss cannot drive phase learning in autoregressive synthesis |
| Vocos-IMBE (reconstruction only) | Trains, unusable audio | Spectral-mean buzz | Reconstruction loss converges to mean immediately |
| Vocos-IMBE (GAN epoch 16) | **Working on synthetic data** | Residual 50Hz flanging | Step-function IMBE frames × per-frame phase prediction |
| Vocos-IMBE (GAN epoch 19+) | Regression | Echo/doubling | Discriminator overfitting |
| Vocos-IMBE on real P25 | Degraded quality | Distribution mismatch + flanging | Encoder mismatch + unresolved flanging |

---

## 4. Suggested Fixes, Ranked by Likely Impact

### Fix 1 (HIGH): Continue GAN + tune discriminator training to avoid echo

**What:** The jump from reconstruction to GAN (epoch 16) fixed the buzz. The degradation at later epochs is discriminator overfitting, not a fundamental limit.

**Evidence:** Epoch 16 sounds good; epoch 19 has echo. The optimal is somewhere in between. Try:
- Reduce GAN learning rate from 1e-4 to 5e-5 for generator
- Longer warmup before enabling generator updates
- R1 gradient penalty on discriminator (weight = 1.0)
- Feature matching loss (L1 on discriminator's internal features) as additional stabilizer

**Experiment:** Train 5 GAN epochs with R1 penalty + feature matching. Compare epochs 1-5 perceptually.

### Fix 2 (HIGH): Train Vocos-IMBE on real P25 data

**What:** The encoder distribution mismatch (sa_sum 2.66×) is the dominant quality issue for the production use case (real radios). Training on real P25 data with properly enhanced clean targets is the fix.

**Evidence:** `project_status.md` documents this as the next required step. Enhancement pipeline (MetricGAN+, 78% acceptance, 35 files/s) is ready. 103K tap files available from trunk-recorder-mqtt.

**Experiment:** Run full enhancement pipeline → convert to binary format → reconstruction pretraining → GAN fine-tuning. Hold out talkgroup for evaluation.

### Fix 3 (MEDIUM): ISTFTHead with UpsampleBridge 2x (architectural, already partially applied)

**What:** The 2x upsample bridge (hop 320→160) is already in the working checkpoint and reduced ACF from 0.43 to 0.09. The remaining flanging may respond to more overlap.

**Evidence:** ACF at frame boundary dropped 5× with 2x upsample. The 4x version diverged, but a different schedule (warm up reconstruction, fine-tune GAN at 4x) might work.

**Experiment:** Try 4x upsample with GAN from start (don't use reconstruction-only pretraining since it diverged in reconstruction phase — GAN may stabilize it).

### Fix 4 (MEDIUM): Hybrid IF head (instantaneous frequency with drift correction)

**What:** The IF head eliminated the 50Hz comb (cepstral peak 0.587→0.021) but caused phase drift because the cumsum is unconstrained. Add periodic phase reset anchored to pitch: at each pitch period boundary, reset the phase to match the expected harmonic relationship.

**Evidence:** IF head demonstrably fixed the comb filter. The drift is a secondary effect of unbounded cumsum. Pitch information (b0) is available in conditioning — use it to constrain cumsum.

**Experiment:** Modify ISTFTHead to accept b0 and use it to reset phase accumulators every N samples at pitch rate. Test whether this eliminates both the comb filter and the drift.

### Fix 5 (LOW): IMBENet GRU with adversarial loss only (no reconstruction)

**What:** Skip reconstruction pre-training for IMBENet entirely. Use a perceptual discriminator (MPD + MRD) from the start, with a tiny regression term (weight=0.1) only for stability.

**Evidence:** The reason FARGAN's adversarial training collapsed was likely because the model was in a bad local minimum from reconstruction pre-training. Training from random init with adversarial loss forces exploration.

**Experiment:** Train 3-GRU IMBENet for 5 epochs with MPD + MRD discriminators, lr=1e-4, no STFT loss. This is a long shot but cheap to try given how fast it trains.

---

## 5. Experiments to Confirm Root Causes

### Confirm IMBENet STFT loss is the limiting factor

**Test:** Compute the STFT loss between the target audio and the **target audio with a fixed random phase offset** applied per STFT frame. If this loss matches or exceeds 0.119, the plateau is explained by phase floor.

```python
# In train.py, after computing target:
import torch
target_noisy_phase = torch.stft(target, n_fft, hop_length, window=window, return_complex=True)
target_noisy_phase = target_noisy_phase * torch.exp(1j * torch.rand_like(target_noisy_phase.angle()) * 2 * torch.pi)
target_random_phase = torch.istft(target_noisy_phase, n_fft, hop_length, window=window)
random_phase_loss = multi_resolution_stft_loss(target, target_random_phase)
print(f"Phase floor: {random_phase_loss.item():.4f}")
```

If this prints ~0.119, we have direct proof that IMBENet's loss plateau is determined by phase information and cannot be reduced by spectral learning alone.

### Confirm Vocos-IMBE flanging frequency

**Test:** Compare cepstral analysis of:
- Vocos-IMBE epoch 16 output (GAN)
- Same test set but with IMBE frames upsampled 2x (nearest neighbor) before feeding to model (should smear the 320-sample comb)

If comb appears at 160 (the new hop size) rather than 320, the artifact source is confirmed as hop-synchronized.

### Confirm encoder mismatch impact

**Test:** Run Vocos-IMBE epoch 16 checkpoint on three inputs:
1. Synthetic LibriSpeech (libimbe-encoded)
2. Real P25 captures (DVSI hardware)
3. Real P25 captures with sa values globally scaled by 2.66× to match libimbe distribution

If audio quality improves in case 3, encoder mismatch is confirmed as the dominant real-P25 quality degradation.

---

## 6. Files and Checkpoints Referenced

| File | Location | Description |
|------|----------|-------------|
| `flanging_final_diagnosis.md` | `~/p25_audio_quality/train/imbenet/` | Detailed quantitative flanging analysis |
| `flanging_fixes_patch.py` | same | UpsampleBridge, ISTFTHeadIF implementations |
| `results_log.md` | `~/p25_audio_quality/train/` | Complete experiment history |
| `imbenet_architecture.md` | `~/p25_audio_quality/train/` | IMBENet GRU design rationale |
| `project_status.md` | `~/p25_vocoder/docs/` | Vocos-IMBE current state, P25 data issues |
| `imbenet_output/imbenet_6.pth` | `~/p25_audio_quality/train/imbenet/` | Best GRU checkpoint (epoch 6, usable only as baseline) |
| `vocos_gan_ep16.pth` | `~/p25_audio_quality/train/imbenet/` | Best Vocos checkpoint (GAN epoch 16) |
| `vocos_gan_ep19.pth` | same | Later GAN checkpoint (echo artifact) |
| `train_eddie.log` | `~/p25_audio_quality/train/imbenet/` | IMBENet plateau evidence |
| `train_aligned.log` | same | Vocos-IMBE training on aligned data |

---

## 7. Key Conclusions

1. **IMBENet (GRU) is architecturally compatible with the task** but STFT loss is the wrong training objective for autoregressive synthesis. Do not train more IMBENet epochs without first switching to a discriminator-only loss.

2. **Vocos-IMBE with GAN is the proven path**. One GAN epoch did more than 15 reconstruction epochs. The architecture works.

3. **The 50Hz flanging has a known root cause and partial fix** (2x upsample). Full elimination requires either the IF head with drift correction, or training longer with GAN to let the discriminator suppress it.

4. **The production blocker is real P25 data mismatch**, not the flanging. Even if flanging were eliminated, the model's responses to real radio features (2.66× lower sa_sum) will be wrong until trained on real P25 data.

5. **No new model architecture is needed.** The working Vocos-IMBE with MPD+MRD GAN at epoch 16 is the foundation. The path forward is data (real P25) and GAN stability (fix echo at later epochs).
