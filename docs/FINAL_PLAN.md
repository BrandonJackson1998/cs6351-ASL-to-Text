# Final ML Plan — ASL Fingerspelling Video → Letter Sequence

**Status:** Draft (2026-05-23)
**Replaces:** PPCA-Mixture-as-decoder (HSMM), synthetic-only CTC, all hand-tuned decoder knobs.

---

## What we've learned (informs everything below)

After multiple iterations, the empirical evidence is clear:

1. **Per-frame static classifiers don't compose well into video pipelines.** PPCA-Mixture hits 98.2% on Kaggle still images but 95.9% LER on FSWild video — distribution shift makes it worthless as an emission model for a sequence decoder.
2. **End-to-end neural models trained on real video data win.** CTC Bi-LSTM at 36.7% test LER on FSWild beats the hand-crafted HSMM by ~60 percentage points.
3. **Hand-tuned decoder knobs (blank floors, span limits, self-loop probabilities) are a trap.** Every knob is a parameter we should be learning from data.
4. **The bottleneck is the emission model's robustness to real-video noise**, not the decoder architecture.

The plan below is informed by these findings.

---

## What gets dropped

These components are cut from the production pipeline. They remain in the repo as artifacts of the exploration, but the final report should treat them as ablations / lessons-learned, not deliverables.

| Component | Reason for cut |
|---|---|
| Synthetic data generator (`synthetic_sequences.py`) | Linear-interp transitions don't match real video; better to use FSWild's 5,400 real sequences |
| HMM decoder (`hmm_decoder.py`) | Required hand-picked blank emission floor — broken by design |
| HSMM decoder (`hsmm_decoder.py` + `hsmm_train.py`) | Forced alignment under Kaggle-trained PPCA produced degenerate spans; demonstrably worse than CTC |
| Pose-change segmenter | No longer needed — CTC doesn't segment |
| Wrist-velocity segmenter | Same |
| Keyframe extractor | Same |
| Segment-vote pipeline | Same |
| string_deltas + segment_vote scripts | Replaced by transcribe_video |

These are not deleted (per user instruction), just not part of the final pipeline.

---

## Final architecture

```
Video
  │
  ▼ MediaPipe Hands per frame, normalized landmarks
T × 63 stream
  │
  ▼ Optional: temporal pooling (causal moving average over W frames)
T × 63 stream (smoother)
  │
  ▼ Bi-LSTM encoder (3 layers, 256 hidden, dropout 0.3)
T × (2 × 256) hidden states
  │
  ▼ Linear head → 27 logits (26 letters + CTC blank)
T × 27 log-probs
  │
  ▼ CTC beam-search decoding with character-level English LM
Letter sequence
```

This is the same Bi-LSTM + CTC architecture we already trained, with three principled improvements:

1. **Larger model + better regularization.** 3-layer 256-hidden Bi-LSTM with dropout 0.3, layer-norm, gradient clipping. Our 2-layer 128-hidden was undersized.
2. **Temporal augmentation at training time.** Random time-warp (resample sequences at 0.7×–1.3× speed), random frame dropout (5%), random landmark jitter. Directly addresses signing-speed variation.
3. **Beam-search with character LM.** Currently we greedy-decode. Adding a small character-level LM trained on English wordlist + FSWild train labels typically drops LER by 3–8 absolute points.

That's it. One model, end-to-end, trained on real data.

---

## Training data

**Sources combined into one training set:**

| Source | Sequences | Notes |
|---|---|---|
| ChicagoFSWild train | 5,429 | Real fingerspelled video, 160 signers, varied angles |
| Kaggle ASL Alphabet | 87k | Used as **single-letter sequences** of length 1, with per-letter labels |

The Kaggle data turns into 87k tiny "fingerspelled words of length 1." This:
- Adds massive per-letter class balance (FSWild has 14 Q's; Kaggle has ~3,000)
- Teaches the model what a clean, isolated letter looks like
- Costs nothing — we already have the landmarks

**Splits:**
- Train: FSWild train + 80% of Kaggle (≈ 75k samples)
- Val: FSWild dev + 10% of Kaggle (≈ 9k)
- Test: FSWild test + 10% of Kaggle (≈ 9k)

**Test set is touched exactly once at the end.**

---

## Augmentation (training only)

Per-sequence, applied with stated probabilities:

| Augmentation | Param | Probability | Why |
|---|---|---|---|
| Time-warp | rate ∈ [0.7, 1.3] | 0.6 | Signing-speed invariance |
| Frame dropout | drop 5% of frames | 0.4 | Noise / detection failures |
| Landmark jitter | σ = 0.015 | 0.6 | MediaPipe noise |
| 3D rotation | ±15° each axis | 0.4 | Camera-angle variance |
| Mirror x | flip | 0.5 | Left-vs-right handedness |

All applied to landmarks, not images. Cheap, deterministic, no re-extraction needed.

---

## Training procedure

1. **Splits:** stratified-ish 60/20/20 within Kaggle, official train/dev/test for FSWild, concat per split.
2. **Optimizer:** Adam, lr=1e-3, cosine annealing.
3. **Loss:** `nn.CTCLoss(blank=0, zero_infinity=True)`.
4. **Batch size:** 32 (sequences padded to max length per batch).
5. **Early stopping:** patience=8 epochs on val LER. Hard cap 60 epochs.
6. **Best-val checkpoint** saved each improvement.
7. **Final test LER** evaluated once on the saved best checkpoint, with greedy *and* beam-search decoders for comparison.

**No hyperparameter tuning on test.** No re-running on the alphabet video to see how it looks. We report what we report.

---

## Beam-search decoder + character LM

Build a 6-gram character LM from the FSWild train labels concatenated with `/usr/share/dict/words`. Use it as a beam-search prior:

```
score(prefix) = log P_ctc(prefix | x) + α · log P_LM(prefix) + β · |prefix|
```

α and β tuned on **dev set only**.

---

## Single-number results to report

For the writeup, two test-set numbers:

| Decoder | Test LER (FSWild) |
|---|---|
| Greedy | (to be measured) |
| Beam + 6-gram LM | (to be measured) |

Plus a confusion-matrix-style breakdown of letter-pair errors and a few qualitative samples.

That's the deliverable. No alphabet-video tuning, no eyeballing, single coherent system.

---

## Phase 2 + 3 stay where they are

- **OpenHands** (word-level WLASL) — already wired, treated as an off-the-shelf baseline integration in the writeup.
- **Phase 3 LLM agent** — separate effort, takes the letter sequence from this CTC model and the gloss list from OpenHands.

This plan only addresses Phase 1c (fingerspelling video → letter sequence).

---

## File-by-file changes

### New / replaced

| File | Purpose |
|---|---|
| `src/preprocessing/temporal_augment.py` | Time-warp, frame dropout (the new augs) |
| `src/preprocessing/combined_dataset.py` | FSWild + Kaggle merged loader |
| `src/models/ctc_lstm.py` | Bigger model: 3 layers × 256 hidden, layer-norm option |
| `src/models/train_ctc_v2.py` | New training script with better augmentation + bigger model |
| `src/models/decode_ctc.py` | Add LM-aware beam search |
| `src/models/char_lm.py` | 6-gram character LM |
| `scripts/transcribe_video.py` | Update to use beam search by default |
| `scripts/evaluate_ctc.py` | Single-shot test eval, prints LER + samples |

### Kept as-is

- `src/models/ppca.py`, `ppca_classifier.py`, `train_ppca.py` — Phase 1 deliverable, intact
- `src/models/wlasl_recognizer.py`, `scripts/recognize_word.py` — Phase 2 OpenHands integration
- `experiments/`, `segments/` — historical artifacts (per user instruction, not deleted)

### Dropped from active pipeline

The HMM/HSMM scripts and segmenters stay on disk but aren't in the new flow.

---

## Implementation order

1. New `combined_dataset.py` with FSWild + Kaggle loading
2. `temporal_augment.py` with time-warp, frame dropout, etc.
3. Bigger `ctc_lstm.py` (3 × 256, layer-norm)
4. `train_ctc_v2.py` with the full augmentation pipeline + early stopping
5. Train (~1–2 hours on CPU)
6. `char_lm.py` + LM-aware beam search in `decode_ctc.py`
7. Tune α, β on dev only
8. `evaluate_ctc.py` — one-shot test evaluation
9. Update `transcribe_video.py` to use the new pipeline
10. Run on `example_videos/alphabet.mp4` exactly once for the report (no tuning)

Estimated time: 4–6 hours including training.
