# IMBE-ASR Model Evaluation

**Date:** 2026-03-25
**Evaluator:** ML Research Agent
**System:** eddie (local GPU workstation)

---

## 1. Current State Assessment

### 1.1 WER Numbers — What We Actually Have

| Model | Dataset | Decoding | WER | CER |
|-------|---------|----------|-----|-----|
| 290M (d=1024, 12L, 16H) | LibriSpeech-IMBE val | Greedy | 6.5% | 1.9% |
| 290M (d=1024, 12L, 16H) | LibriSpeech-IMBE val | Beam+5gram KenLM | **1.9%** | **0.7%** |
| 48.6M base (d=512, 8L, 8H) | LibriSpeech-IMBE val | Greedy | 18.9% | 6.8% |
| 48.6M P25-finetuned (epoch 9) | LibriSpeech-IMBE val | Greedy | 19.2% | — |
| 48.6M P25-finetuned | Real P25 .tap files | Greedy | **Unknown (no ground truth)** | — |
| Original 9.2M (d=256, 6L) | LibriSpeech-IMBE val | Greedy | 40.6% | — |
| Original 9.2M | LibriSpeech-IMBE val | Beam+LM | ~28-30% | — |

**Key observations:**

1. **The 290M model is dramatically better.** 6.5% greedy vs 18.9% for the 48.6M — a 3× reduction in WER. With beam+LM decoding, 1.9% WER is near state-of-the-art for LibriSpeech (standard Conformer-CTC on mel features gets ~2-3% with LM). This proves the IMBE→text approach works.

2. **Beam search + KenLM is transformative.** The 290M model drops from 6.5% → 1.9% (3.4× improvement). The same relative gain on the 48.6M base would project to roughly 5-7% WER with beam+LM (not yet measured, but the decode infrastructure exists at `src/decode.py`).

3. **P25 fine-tuning barely hurts LibriSpeech.** 18.9% → 19.2% (0.3% regression) — well within noise. But P25 transcriptions are "much more readable" per the results log, with corrected unit names (METIC→MEDIC) and dispatch structure.

4. **No quantitative WER on real P25.** We cannot measure true P25 WER because there are no ground-truth transcripts. The `results.txt` is qualitative only.

### 1.2 Base (48.6M) vs Large (290M)

The 290M model trained for 30 epochs on the expanded ~1220h dataset (`data_expanded.yaml`) on an H100/Sarah:

| Metric | 48.6M (eddie, ep 25) | 290M (sarah, ep 30) | Delta |
|--------|---------------------|---------------------|-------|
| Greedy WER | 18.9% | 6.5% | -12.4pp |
| Greedy CER | 6.8% | 1.9% | -4.9pp |
| Beam+LM WER | Not measured | 1.9% | — |
| Beam+LM CER | Not measured | 0.7% | — |
| Parameters | 48.6M | 290M | 6× |

The training curve for the 1024d model (`training_curve_1024d.txt`) shows WER was still improving at epoch 30 (14.7% greedy at convergence, then eval with proper beam yields 6.5%). The curve went from 55.2% at epoch 2 to 14.7% at epoch 30 — steady improvement throughout, no plateau.

The sweep results (`sweep_results_5090.txt`) confirm **depth matters more than width**: d=768/12L/ff=4 (163.5M) achieved 30.6% WER on a 15% data subset in 5 epochs. The smaller configs (d=256, d=384) mostly crashed or achieved 60-100% WER. The optimal architecture identified by Bayesian sweep is d=1024, 12 layers, ff_mult=4.

### 1.3 Greedy vs Beam Search + LM

For the 290M model:
- **Greedy: 6.5% WER → Beam+LM: 1.9% WER** — a 70% relative error reduction.

Typical errors fixed by beam+LM (from eval output):
- "POSICIAN" → "POSITION" (phonetic misspelling corrected by LM)
- "PASEBOARD" → "PASTEBOARD" (missing T restored)
- "GARDEN COURT" → "GARDENCOURT" (word boundary fixed)
- "ILLUSIONS" → "ALLUSIONS" (similar-sounding word corrected)
- "WHAT IS" → "WHAT'S" (contraction handling)

The LM is a 5-gram KenLM trained on LibriSpeech text + P25 transcripts. Files exist at `data/lm/5gram.bin` with `unigrams.txt` (28,375 words). The decoder uses pyctcdecode with alpha=0.5, beta=1.5, beam_width=100.

**Critical gap: beam+LM has NOT been evaluated on the 48.6M base model.** This is the model actually deployed for P25 inference. Given the 70% relative reduction on the 290M, we'd expect substantial improvement on the 48.6M too.

### 1.4 Inference Speed

**Raspberry Pi 5 (4GB, aarch64) benchmarks** — this is the deployment target:

| Runtime | Model | 10s call | RTF | RAM |
|---------|-------|----------|-----|-----|
| C + ONNX fp32 | 48.6M | 660ms | 0.066x | 285MB |
| Python ONNX fp32 | 48.6M | 649ms | 0.065x | 285MB |
| C + ONNX uint8 | 48.6M | 788ms | 0.079x | — |
| PyTorch fp32 | 48.6M | 800ms | 0.080x | 995MB |
| C + ONNX uint8 | 290M | 2777ms | 0.278x | ~1.3GB |
| Python ONNX int8 | 290M | 3507ms | 0.351x | 535MB |
| PyTorch fp32 | 290M | OOM | — | >4GB |

**eddie (GPU workstation) benchmarks** (from CLAUDE.md): 5.8ms GPU / 49ms CPU for a 10s call (200× real-time on CPU, ~1700× on GPU).

**Key takeaways:**
- Pi 5 handles the 48.6M model at 15× real-time (C+ONNX). Plenty fast for live P25.
- The 290M model fits on Pi 5 as int8 ONNX (2.8s for 10s call, 3.6× real-time). Usable but tight.
- On eddie, everything is trivially fast. CPU inference at 49ms/10s is more than sufficient.
- The C inference engine (70KB binary, no Python) matches Python ONNX performance.

---

## 2. Known Issues & Root Causes

### 2.1 IMBE Vocoder Artifacts (from imbenet-artifact-analysis.md)

The artifact analysis covers the **vocoder reconstruction** path (IMBE→audio→ASR), NOT the direct IMBE→text path. However, the findings are relevant because the pseudo-labeling pipeline decodes IMBE→audio→Whisper/Qwen3 to generate P25 training labels.

**Summary of vocoder issues:**

1. **IMBENet (GRU) — dead end.** STFT loss plateaus at 0.119 from epoch 1. Cannot learn phase. Outputs spectral-mean buzz. All 6 GAN configurations collapsed.

2. **Vocos-IMBE (ISTFTHead) — partially working.** GAN epoch 16 produces usable audio but with a **50Hz comb filter/flanging artifact** caused by non-overlapping 20ms IMBE frames creating step-function feature boundaries. The ISTFTHead predicts phase independently per frame, causing constructive/destructive interference at frame boundaries.

3. **Encoder distribution mismatch — the real blocker.** Training data uses libimbe (software encoder); real P25 uses DVSI hardware. Measured differences:
   - f0: 174.5 Hz (libimbe) vs 217.7 Hz (real P25) — 0.80× ratio
   - sa_sum: 823.7 (libimbe) vs 310.1 (real P25) — **2.66× ratio**
   
   This means the ASR model trained on LibriSpeech-IMBE sees systematically different spectral energy distributions from real P25 radio.

### 2.2 Distribution Mismatch: LibriSpeech-IMBE vs Real P25

This is the **single largest quality concern** for the direct IMBE→text ASR path. The mismatch has multiple dimensions:

| Dimension | LibriSpeech-IMBE | Real P25 | Impact |
|-----------|-----------------|----------|--------|
| Encoder | libimbe (software) | DVSI chip (hardware) | Different sa distributions (2.66× energy gap) |
| Audio quality | Studio recording | RF noise, multipath, channel distortion | FEC errors, spectral noise |
| Content | Literary narration | Dispatch, casual radio, 10-codes | Vocabulary mismatch |
| Speaking style | Clear, enunciated | Rapid, clipped, background noise | Prosodic mismatch |
| Silence patterns | Clean silence between utterances | P25 signaling frames (LICH, HDU, LDU) | Already handled (silence stripping) |
| Channel effects | None | AGC, IMBE quantization artifacts | Unknown impact magnitude |

The encoder mismatch alone (2.66× sa_sum) means the z-score normalization (computed from LibriSpeech-IMBE training stats saved as `stats.npz`) may be wrong for real P25 input. The mean/std for spectral amplitude dimensions [2:58] will be off by a large factor.

**Practical impact:** The model likely still gets the phonetic content approximately right (it learned relative spectral patterns, not absolute magnitudes), but confidence scores will be miscalibrated, and edge-case decoding decisions will be worse. The P25 fine-tuning (19,787 pseudo-labeled NPZ files in `data/p25_labeled/`) partially addresses this by exposing the model to real P25-encoded features, but 20 hours of P25 data vs 1,220 hours of LibriSpeech doesn't shift the distribution much.

### 2.3 Practical Impact on Real-World Quality

Without ground-truth transcripts, quality assessment is qualitative. From `results.txt`:
- P25-finetuned model produces "readable fire dispatch transcriptions"
- Recognizes "BATTALION 60 ENGINE 62 MEDIC 61"
- Improvements: METIC→MEDIC, QUIN MEKITO→QUINT 62, HOWER→POWER/TOWER
- Remaining issues: "WHO WHO WHO" preamble noise, number formatting

The gap between 1.9% WER on LibriSpeech-IMBE (with the 290M model) and "sometimes gets unit names wrong" on real P25 suggests the real-P25 WER is likely **30-50%+** for the 48.6M model (greedy, no LM). This is usable for monitoring/triage but not reliable transcription.

---

## 3. What Can Be Tested Now (on eddie)

### 3.1 Live P25 Test Stack

The system has:
- trunk-recorder capturing live P25 radio
- symbolstream plugin sending IMBE frames via TCP (§12 v1 format currently, v2 spec ready)
- `src/inference.py --watch` mode for automatic transcription of new .dvcf files
- `src/live.py` for socket-based live transcription

### 3.2 Metrics to Capture

**Beyond "does it produce text":**

1. **Transcription rate:** What fraction of calls produce non-empty transcriptions? (Track total .dvcf files vs transcribed outputs)

2. **Character-level confidence:** Extract per-frame CTC log-probabilities from `model.forward()`. Compute:
   - Mean max log-prob per frame (higher = model is more certain)
   - Fraction of frames where max prob < threshold (uncertainty frames)
   - Per-utterance average confidence

3. **Output statistics:** 
   - Average words per transcription
   - Vocabulary distribution (are we seeing expected P25 terms?)
   - Repeated-token frequency (stuttering like "THE THE THE" indicates CTC collapse)
   - Blank-frame ratio (what fraction of output is CTC blank?)

4. **Timing metrics:**
   - Inference latency per call (already logged in --watch mode)
   - Call duration distribution
   - Frames per call after silence stripping

5. **FEC error correlation:** From symbolstream metadata (errs field per frame), correlate FEC error count with transcription confidence. Calls with errs≥3 should have systematically lower quality.

### 3.3 A/B Comparison: IMBE ASR vs Qwen3 ASR

**Yes, this is feasible and should be done.** The pseudo-labeling pipeline already has the pieces:

1. For each .dvcf file, decode IMBE→audio via libimbe (8kHz PCM)
2. Send audio to Qwen3-ASR server (`http://localhost:8765/v1/audio/transcriptions`)
3. Run IMBE-ASR on the same .dvcf file directly
4. Compare outputs

**Comparison script outline:**
```bash
# For each new .dvcf file:
# 1. IMBE-ASR: python -m src.inference --dvcf-file $FILE --checkpoint ...
# 2. Qwen3-ASR: decode IMBE→WAV, POST to Qwen3 server
# 3. Log both transcriptions side-by-side
```

This gives us:
- Relative quality comparison (which model produces more coherent output?)
- Pseudo-ground-truth: Qwen3 on decoded audio is an independent ASR system; agreement between the two is a signal of correctness
- Disagreement analysis: where do they diverge? (noise, domain terms, numbers)

### 3.4 Quality Without Ground Truth

**Semi-supervised quality estimation approaches:**

1. **Multi-ASR consensus:** Already built into `pseudo_label.py` (Whisper + Qwen3). When both agree within 30% edit distance, the transcript is likely correct. Consensus rate = proxy for quality.

2. **LLM-as-judge:** Send IMBE-ASR and Qwen3-ASR transcripts to an LLM. Ask: "Which is more coherent? Rate each 1-5." This is cheap and gives a quality distribution.

3. **Keyword spotting accuracy:** Compile a list of expected P25 terms (known talkgroup names, unit IDs, common dispatch phrases). Measure how often IMBE-ASR outputs contain these terms vs Qwen3-ASR.

4. **Manual spot-check:** Pick 50-100 random calls. Listen to decoded audio. Score IMBE-ASR transcript on a 1-5 scale. This is the gold standard and takes ~2-3 hours.

---

## 4. Quick Wins (No Retraining Required)

### 4.1 Beam Search + LM on 48.6M Model

**Status: Ready to deploy. Infrastructure exists.**

- KenLM 5-gram binary: `data/lm/5gram.bin` ✓
- Unigrams file: `data/lm/unigrams.txt` (28,375 words) ✓
- Beam decoder: `src/decode.py` with `BeamDecoder` class ✓
- pyctcdecode integration: complete ✓

**Expected improvement:** The 290M model saw 70% relative WER reduction (6.5% → 1.9%). The 48.6M model at 18.9% greedy could plausibly drop to **8-12% WER** with beam+LM. Even a conservative 40% relative reduction would give ~11% WER.

**Effort:** ~1 hour. Run `python -m src.eval checkpoints/eddie_512d/best.pth --beam --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt`.

**For P25 specifically:** The LM was built from LibriSpeech text + some P25 transcripts (`data/lm/p25_text.txt`, `p25_text_weighted.txt`). A domain-specific LM would help more (see §6).

**Caveat:** Beam search is slower. On Pi 5, pyctcdecode beam search (width=100) could add 2-5× latency. May need reduced beam width (10-20) for real-time.

### 4.2 Better Silence/Noise Filtering

**Current state:** `_strip_silence_frames()` in `inference.py` removes zero-energy frames by decoding through libimbe and checking `sum(abs(raw_params[:, 2:58])) > 0`. This catches P25 signaling but not:

- Low-energy noise frames (radio hiss between transmissions)
- Very short bursts (key clicks, squelch tails)
- FEC-corrupted frames that produce garbage spectral content

**Quick improvements:**
1. **Energy threshold instead of zero check:** Replace `energy > 0` with `energy > threshold` (calibrate from real P25 data, maybe 10th percentile of voice frame energy).
2. **FEC error filtering:** Skip frames with `errs ≥ 4` (per IMBE FEC guidance in symbolstream spec).
3. **Minimum duration:** Reject calls with < 0.5s of voice frames after filtering (currently `min_frames=10` = 200ms, probably too short).

**Effort:** ~2 hours. Modify `_strip_silence_frames()` and test on existing .dvcf files.

### 4.3 Post-Processing

The model outputs raw uppercase text with no punctuation, no formatting, no numbers. Post-processing would dramatically improve readability:

1. **Inverse text normalization (ITN):** Convert spelled-out numbers to digits. "SIXTY TWO" → "62", "ONE HUNDRED TWENTY THREE MAIN STREET" → "123 MAIN STREET". NeMo ITN or a simple regex-based approach.

2. **Punctuation restoration:** Add periods at sentence boundaries, commas for lists. Can use a small punctuation model or rule-based approach for radio dispatch (utterances are typically short, imperative).

3. **Capitalization:** Convert to title case or sentence case. Keep known proper nouns (unit names, street names) capitalized.

4. **P25-specific formatting:**
   - "TEN FOUR" → "10-4"
   - "TALKGROUP NINE ONE SEVEN ZERO" → "TG 9170"
   - Known unit patterns: "ENGINE SIXTY TWO" → "Engine 62"
   - Address formatting: "ONE TWO THREE MAIN STREET" → "123 Main St"

**Effort:** ~1-2 days for a useful first version. Regex-based, no ML required.

### 4.4 Confidence Thresholds

**Reject low-confidence transcriptions** to avoid garbage output:

1. Extract per-frame max log-probability from CTC output
2. Compute utterance-level mean confidence
3. Set threshold (calibrate on labeled data): reject if mean confidence < X
4. Return "[LOW CONFIDENCE]" or "[UNINTELLIGIBLE]" instead of garbage text

Implementation in `inference.py` is straightforward — the `log_probs` tensor is already available. Add:
```python
frame_confidence = log_probs.max(dim=-1).values.mean().item()
if frame_confidence < threshold:
    return "[UNINTELLIGIBLE]"
```

This is especially valuable for P25 where many calls are encrypted (output = random garbage) or heavily corrupted.

**Effort:** ~1 hour to implement, ~2 hours to calibrate threshold.

### 4.5 Deploy 290M Model on eddie

The 290M model achieves 1.9% WER with beam+LM — **vastly** better than the 48.6M. On eddie's GPU, it runs at 5.8ms for a 10s call. There's no reason not to use it on eddie for the live test stack.

The checkpoint exists at `checkpoints/sarah_1024d/best.pth` (or `h100_1024d/best.pth`). SafeTensors and ONNX exports also available.

**Effort:** 10 minutes. Change the checkpoint path in the inference command.

---

## 5. Training Improvements Needed

### 5.1 More Real P25 Training Data

**Current state:** ~19,787 pseudo-labeled NPZ files in `data/p25_labeled/`, likely representing ~20 hours of P25 audio. This is against ~1,220 hours of LibriSpeech/TEDLIUM/GigaSpeech.

**How much more:** The standard fine-tuning rule of thumb is 10-100 hours of in-domain data to meaningfully shift a large model. For a 48.6M model already trained on 1,220h, we'd want:
- **50-100 hours** of pseudo-labeled P25 for good domain adaptation
- **200+ hours** to make P25 the dominant distribution
- **500+ hours** for a P25-primary model

**Sources:**
1. **trunk-recorder archive:** The artifact analysis mentions 103K .tap files from trunk-recorder-mqtt. At ~5s average per call, that's ~140 hours — enough for meaningful fine-tuning.
2. **Live collection:** The test stack is continuously recording. A week of active monitoring could yield 10-50 hours depending on talkgroup activity.
3. **Public P25 datasets:** Broadcastify archives, RadioReference audio — would need IMBE-encoding.
4. **Multi-system collection:** Different P25 systems (different DVSI hardware, different RF environments) for robustness.

### 5.2 Pseudo-Labeling Pipeline

**Status: Built and working.** `scripts/pseudo_label.py` uses Whisper + Qwen3-ASR dual-labeling:
- Decodes .tap → WAV via libimbe
- Sends to both Whisper (localhost:8766) and Qwen3-ASR (localhost:8765)
- Accepts consensus transcripts (edit distance < 30%)
- Flags disagreements

**Quality concerns:**
1. **The pipeline transcribes reconstructed audio, not raw IMBE.** The vocoder artifacts (50Hz comb, encoder mismatch) degrade the audio that Whisper/Qwen3 see. This means pseudo-labels have systematic errors correlated with P25 audio quality.
2. **Low-quality calls get wrong labels.** If Whisper and Qwen3 both get it wrong the same way (e.g., both mishear a noisy word), the model learns the wrong label.
3. **Selection bias:** The 30% edit distance threshold means only "easy" calls get labeled. Hard calls (noisy, short, domain-specific) are filtered out — but those are exactly the ones we need to train on.

**Improvements:**
- Use the 290M IMBE-ASR model as a third voter (it sees raw IMBE, not reconstructed audio)
- Lower the agreement threshold for longer calls (more context = less ambiguity)
- Manual review of a random sample to estimate label error rate

### 5.3 Domain-Constrained Language Model

**P25 radio IS general conversational English.** Per CLAUDE.md: "P25 covers law enforcement, fire, EMS, public works, transit, schools, bus drivers, ODOT workers -- full breadth of everyday English conversation." It is NOT limited to dispatch codes. There's significant casual chatter, and the roadmap includes DMR which covers commercial and amateur radio — even more conversational. A constrained dispatch-only LM would hurt on this content. That said, P25 does have SOME domain-specific patterns worth boosting:

- **Unit identifiers:** "Engine 62", "Medic 41", "Battalion 60", "Ladder 7"
- **Addresses:** "123 Main Street", "Interstate 75 southbound"
- **Dispatch phrases:** "Respond to", "En route", "On scene", "Clear the call"
- **10-codes:** "10-4", "10-8", "Signal 7" (varies by jurisdiction)
- **Phonetic alphabet:** "Adam", "Boy", "Charlie" (or NATO: "Alpha", "Bravo")
- **Status codes:** "Code 3", "Priority 1", "Non-emergency"

A domain-aware LM should boost these patterns while preserving general English fluency. DO NOT over-constrain — the model needs to handle everything from "ENGINE 62 RESPOND TO 450 MAIN STREET" to "hey did you see the game last night" on the same system.

### 5.4 Fine-Tuning the 290M Model on P25

The 290M model is the clear performance winner. Fine-tuning it on P25:

**Strategy:**
1. Start from `checkpoints/sarah_1024d/best.pth` (or `h100_1024d/best.pth`)
2. Use `scripts/finetune_p25.py` with `--base-fraction 0.3` (70% P25, 30% base data)
3. Lower LR: 1e-5 to 5e-5 (the 48.6M used 5e-5)
4. 5-10 epochs should be sufficient
5. Multi-GPU via torchrun on eddie (2× GPU if available) or rent an H100 spot instance

**Expected outcome:** The 290M model at 1.9% WER on LibriSpeech should maintain ~2-3% WER post-fine-tuning while substantially improving on real P25. The P25 results.txt improvements seen with the 48.6M fine-tuning would be even more dramatic with the larger model.

**GPU requirements:** The 290M model needs ~8GB+ VRAM for inference, ~24GB+ for training with batch size 16. An A100 or H100 is ideal. Eddie's GPU should work if it's a 3090/4090 (24GB) or better.

---

## 6. Language Model Options

### 6.1 Current KenLM N-gram

**What we have:** A 5-gram KenLM model built from LibriSpeech text + P25 transcripts.

**Files:**
- `data/lm/5gram.arpa` — ARPA format
- `data/lm/5gram.bin` — KenLM binary (fast)
- `data/lm/combined_lm_text.txt` — combined training corpus
- `data/lm/librispeech_normalized.txt` — LibriSpeech text
- `data/lm/p25_text.txt` — P25 transcript text
- `data/lm/p25_text_weighted.txt` — P25 text with upsampling weight
- `data/lm/unigrams.txt` — 28,375 unigram vocabulary

**Is it good enough?** For LibriSpeech, yes — 6.5% → 1.9% WER is excellent. For P25, probably not. The LM is dominated by LibriSpeech literary text (28K+ words from novels/lectures). P25 dispatch patterns are underrepresented.

### 6.2 Domain-Specific Dispatch LM

**Would it help?** Almost certainly yes, and potentially dramatically.

A radio-aware LM trained on:
- Existing P25 pseudo-labels (~20h of transcripts)
- Manual transcripts from spot-checking
- Synthetic dispatch text generated by LLM (cheapest source of large-scale domain text)
- Broadcastify transcript archives

Could learn:
- "ENGINE" is almost always followed by a number
- "RESPOND TO" is followed by an address
- "10-" is followed by a 1-2 digit code
- Unit ID patterns specific to the monitored system

**Expected impact:** For domain-specific terms (unit IDs, addresses, codes), a radio-aware LM could reduce errors by 50%+. General conversational content also benefits from radio speech patterns (short utterances, PTT cadence, acknowledgment patterns). Critical: the LM must handle general English well since DMR/amateur radio is full conversational speech.

**Implementation:** The `scripts/build_lm.py` infrastructure already supports P25 text input. Build a separate P25-only KenLM and interpolate with the general LM:

```python
# In decode.py, modify BeamDecoder to support interpolated LMs
# pyctcdecode doesn't natively support LM interpolation, but you can:
# 1. Build a combined corpus with P25 text heavily upweighted (10-50×)
# 2. Train a single KenLM on the weighted corpus
# 3. Use hotwords for high-frequency P25 terms
```

The `load_hotwords()` function in `decode.py` already supports hotword boosting (weight=10.0). A hotwords file of unit IDs, talkgroup names, and common dispatch terms is a quick win.

**Effort:** 
- Hotwords file: 1-2 hours (compile from known system data)
- P25-weighted KenLM: 2-4 hours (generate synthetic dispatch text, retrain LM)
- Full dispatch LM with interpolation: 1-2 days

### 6.3 Constrained Decoding

**Talkgroup-specific vocabulary:** If we know which talkgroup a call is on (we do — it's in the .dvcf metadata), we can apply talkgroup-specific constraints:

- Fire dispatch TG: boost "ENGINE", "LADDER", "MEDIC", "BATTALION", address terms
- Law enforcement TG: boost "UNIT", "SUSPECT", "VEHICLE", "WARRANT"
- EMS TG: boost "PATIENT", "TRANSPORT", "ALS", "BLS"

Implementation: Pass talkgroup ID to `BeamDecoder`, select appropriate hotwords list.

**Unit ID patterns:** Known unit number ranges per system. "Engine" is followed by 1-99. "Medic" is followed by 1-99. These could be encoded as FST constraints in the decoder, but pyctcdecode doesn't support arbitrary FST. Hotword boosting is the practical approach.

### 6.4 CTC + Attention Hybrid vs Pure CTC

**Current architecture:** Pure CTC (Conformer encoder → Linear → log_softmax).

**Hybrid CTC + Attention** adds an autoregressive attention decoder alongside CTC. Benefits:
- Better handling of word boundaries (attention learns segmentation)
- Implicit language model in the decoder
- Can optionally do joint CTC+attention decoding

**Is it worth it?** For this project, probably not yet:
1. CTC + external KenLM already achieves 1.9% WER — hard to beat
2. Attention decoders add latency (autoregressive generation)
3. Implementation complexity (need to modify model.py significantly)
4. The bottleneck is P25 domain adaptation, not architecture

If CTC + KenLM proves insufficient after P25 fine-tuning + domain LM, then a hybrid approach could be explored. But it's a large effort for uncertain gain.

---

## 7. Recommendations (Prioritized)

### Priority 1: Deploy 290M model + beam search on eddie (Impact: HIGH, Effort: LOW)

**What:** Switch eddie's live test stack from the 48.6M P25-finetuned model to the 290M model with beam+LM decoding.

**Why:** The 290M model is 3.4× better on LibriSpeech-IMBE (1.9% vs ~7% estimated for 48.6M+LM). On eddie's GPU, inference is trivially fast (5.8ms for 10s). This is a configuration change, not a code change.

**Steps:**
1. Update checkpoint path to `checkpoints/sarah_1024d/best.pth`
2. Enable beam search in inference (add `--beam --lm-path data/lm/5gram.bin`)
3. Verify on a handful of .dvcf files

**Expected: ~3-5× better transcription quality. 10 minutes of work.**

### Priority 2: Measure 48.6M beam+LM WER (Impact: MEDIUM, Effort: LOW)

**What:** Run the beam+LM evaluation on the 48.6M base model to fill in the missing number.

**Why:** This tells us the actual gap between our deployed model and the 290M. If 48.6M+LM ≈ 10% WER, it's "good enough" for Pi 5 deployment. If it's still 15%+, we need the 290M on Pi.

**Steps:** `python -m src.eval checkpoints/eddie_512d/best.pth --beam --lm-path data/lm/5gram.bin --unigrams data/lm/unigrams.txt`

**Expected: ~8-12% WER. 1 hour of work (mostly inference time).**

### Priority 3: A/B comparison vs Qwen3-ASR on live P25 (Impact: HIGH, Effort: MEDIUM)

**What:** Build a comparison script that runs both IMBE-ASR and Qwen3-ASR (via reconstructed audio) on the same .dvcf files, logs both outputs.

**Why:** This is the only way to get a quality signal on real P25 without manual transcription. Qwen3-ASR on 8kHz decoded audio is the "traditional" approach — measuring how IMBE-ASR compares tells us if the direct codec→text approach actually works better.

**Steps:**
1. Write a comparison script (modify --watch mode to also run Qwen3)
2. Run for 24-48 hours on live P25
3. Compute agreement rate, divergence analysis
4. LLM-judge quality comparison on disagreements

**Expected: 1-2 days to build + 2 days to collect data. Critical for go/no-go decision.**

### Priority 4: Confidence thresholds + noise filtering (Impact: MEDIUM, Effort: LOW)

**What:** Implement utterance-level confidence scoring, reject low-confidence outputs, improve silence/noise filtering.

**Why:** Reduces garbage output (encrypted calls, heavily corrupted calls, squelch tails). Improves perceived quality even if WER doesn't change.

**Steps:** Modify `inference.py` per §4.4 and §4.2.

**Expected: Cleaner output, fewer false transcriptions. ~4 hours of work.**

### Priority 5: P25-specific hotwords + weighted LM (Impact: MEDIUM, Effort: LOW)

**What:** Build a hotwords file with known P25 terms. Optionally rebuild KenLM with P25 text heavily upweighted.

**Why:** Low-hanging fruit for domain adaptation without retraining. The `BeamDecoder` already supports hotwords with `hotword_weight=10.0`.

**Steps:**
1. Compile hotwords: unit IDs, talkgroup names, common dispatch terms
2. Pass to `BeamDecoder(hotwords=load_hotwords("data/lm/p25_hotwords.txt"))`
3. Optionally rebuild LM: `python scripts/build_lm.py --p25-weight 20`

**Expected: 10-30% relative improvement on P25-specific terms. 4-8 hours of work.**

### Priority 6: Fine-tune 290M on P25 data (Impact: HIGH, Effort: MEDIUM)

**What:** Run `scripts/finetune_p25.py` with the 290M checkpoint as base.

**Why:** The 48.6M fine-tuning showed meaningful P25 improvement with only 0.3% LibriSpeech regression. The 290M model should benefit even more from P25 exposure.

**Prerequisites:** Need more P25 pseudo-labeled data (Priority 7). The current 20 hours is thin for a 290M model.

**Steps:**
1. Generate more pseudo-labels (see Priority 7)
2. `torchrun --nproc_per_node=2 scripts/finetune_p25.py --checkpoint checkpoints/sarah_1024d/best.pth --p25-dir data/p25_labeled --epochs 5 --lr 2e-5`
3. Evaluate on both LibriSpeech and live P25

**Expected: Near-SOTA LibriSpeech WER + significantly better P25. 1-2 days (GPU time + evaluation).**

### Priority 7: Scale P25 pseudo-labeling (Impact: HIGH, Effort: MEDIUM)

**What:** Process the full 103K .tap file archive through the pseudo-labeling pipeline.

**Why:** 20 hours of P25 data isn't enough. 100+ hours would make P25 fine-tuning much more effective.

**Steps:**
1. Ensure Whisper server (localhost:8766) and Qwen3-ASR server (localhost:8765) are running
2. Run `python3 scripts/pseudo_label.py --tap-dir /path/to/full/archive --output-dir data/p25_labeled_v2 --workers 8`
3. Quality audit: spot-check 100 random outputs

**Expected: 50-100 hours of pseudo-labeled P25 data. 1-2 days of compute.**

### Priority 8: Post-processing pipeline (Impact: MEDIUM, Effort: MEDIUM)

**What:** Build inverse text normalization + punctuation + P25-specific formatting.

**Why:** Raw uppercase text with no formatting is hard to read. Formatting dramatically improves usability even at the same WER.

**Expected: Much more readable output. 1-2 days of work.**

### Priority 9: Domain-specific KenLM (Impact: MEDIUM, Effort: MEDIUM)

**What:** Generate synthetic dispatch text via LLM, build P25-focused KenLM, test interpolation.

**Expected: 20-40% relative improvement on dispatch-specific content. 2-3 days.**

### Priority 10: Explore CTC + attention hybrid (Impact: UNKNOWN, Effort: HIGH)

**What:** Add an attention decoder to the Conformer encoder for joint CTC+attention decoding.

**Why:** Only if priorities 1-9 leave WER unsatisfactory. Current architecture with beam+LM is already at 1.9% on clean data.

**Expected: Marginal improvement. 1-2 weeks of development.**

---

## Summary

The IMBE→text ASR approach is **validated and working**. The 290M model at 1.9% WER with beam+LM on LibriSpeech-IMBE proves that direct codec parameter→text recognition is viable and competitive with standard mel-spectrogram ASR.

The immediate bottleneck is **not the model architecture** — it's deployment and domain adaptation:

1. **Deploy the 290M model on eddie now** (10 minutes, massive quality gain)
2. **Measure what we actually have** (beam+LM on 48.6M, A/B vs Qwen3)
3. **Adapt to P25 domain** (more data, better LM, hotwords, fine-tuning)
4. **Improve user experience** (confidence filtering, post-processing, formatting)

The engineering path from "research prototype" to "useful P25 transcription system" is clear and mostly involves work we already have infrastructure for.
