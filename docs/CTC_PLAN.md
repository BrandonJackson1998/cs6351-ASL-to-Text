# Fingerspelling via CTC — Plan

**Status:** Draft (2026-05-21)
**Replaces:** segment-then-classify pipeline for fingerspelling
**Phase:** 1c (after PPCA baseline)

---

## Why this approach

The segmentation-then-classify pipeline (current `string_deltas` and `segment_vote`)
hits a hard ceiling because the segmenter has to commit to letter boundaries
before the classifier ever sees the full picture. When boundaries land mid-
transition, no amount of voting recovers the missed letter.

CTC (Connectionist Temporal Classification, Graves et al. 2006) was invented
for exactly this problem in speech recognition: variable-length input
sequences (frames) → variable-length output sequences (letters), with
**no per-frame alignment required during training.**

The model emits a probability distribution over letters + a "blank" token
for every frame. CTC marginalizes over all valid alignments during training
(via dynamic programming) and uses beam search to decode at inference time.
The **boundary detection becomes part of the model** — implicit, learned,
not a brittle preprocessing step.

This also lines up exactly with the proposal's Phase 2 plan: "extract
per-frame MediaPipe landmarks → train an LSTM." Switching the loss from
cross-entropy to CTC is the only architectural change.

---

## Architecture

```
Video
  │
  ▼
MediaPipe per-frame landmarks  ──►  T × 63 floats (length T = ~fps × duration)
  │
  ▼
Optional: PPCA  (T × 63 → T × 12)   [denoising, reuse trained PPCA]
  │
  ▼
Bi-LSTM encoder (2 layers, hidden=128)
  │
  ▼
Linear → softmax over (26 letters + 1 blank) per frame
  │
  ▼ CTC loss vs. ground-truth letter sequence (training)
  │ Beam-search decode (inference)
  ▼
Letter sequence: ['A', 'B', 'C', ...]
```

**Why Bi-LSTM, not Transformer:** smaller training set, faster to converge,
matches proposal language. Stretch goal: swap for a small Transformer
encoder if results plateau.

**Why include PPCA in front:** the trained PPCA from Phase 1b already
denoises 63D landmarks to ~12D; this gives the LSTM a cleaner signal and
fewer parameters to learn. Optional — can ablate.

---

## Training Data Strategy

We need **fingerspelled word sequences** with letter-level labels. Building
those:

### Source 1: Synthetic concatenation from ASL Alphabet (primary)

Stitch together random sequences of letters from the existing ASL Alphabet
landmark dataset to create artificial "spelled words":

1. Sample word lengths uniformly from {2, 3, 4, 5, 6, 7, 8} letters
2. For each letter slot, randomly sample one landmark vector from that
   letter's training pool
3. Insert ~5-15 frames of "transition" between letters, where transition
   landmarks are linearly interpolated between letter A's landmarks and
   letter B's, with added noise
4. Insert random "hold" durations (5-30 frames) on each letter
5. Augment as before: jitter, rotation, mirror

This gives us **infinite training data** with exact letter-level labels.
Generate ~50k synthetic sequences.

**Risk:** synthetic transitions don't perfectly match real video transitions.
Mitigation: validate on Source 2.

### Source 2: Real video clips (validation/finetune)

Manually transcribe a small set of fingerspelled video clips (~50 short
videos, 20-50 minutes of work). Use these for validation and optional
fine-tuning. We already have `example_videos/alphabet.mp4` (a-z) and
others to start with.

### Source 3: ChicagoFSWild dataset (stretch)

[ChicagoFSWild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) — ~7,000
fingerspelled video clips with letter-level labels. Free for academic use,
already preprocessed. Drop-in replacement for synthetic data if synthetic
underperforms. ~1.6 GB.

---

## File-by-File Changes

### New files

| File | Purpose |
|------|---------|
| `src/models/ctc_lstm.py` | Bi-LSTM + CTC head model definition |
| `src/models/train_ctc.py` | Training loop (CTC loss, padding/packing, val WER) |
| `src/models/decode_ctc.py` | Greedy + beam-search decoder |
| `src/preprocessing/synthetic_sequences.py` | Generate fingerspelled sequences from ASL Alphabet landmarks |
| `src/preprocessing/video_landmark_stream.py` | Extract a length-T landmark stream from a video (no segmentation) |
| `scripts/transcribe_video.py` | End-to-end: video → letter sequence using CTC model |
| `tests/test_synthetic_sequences.py` | Unit test for synthetic data generation |
| `tests/test_ctc_decode.py` | Sanity-check decoder on toy sequences |

### Modified files

| File | Change |
|------|--------|
| `Makefile` | New targets: `make synth-data`, `make train-ctc`, `make transcribe-video` |
| `requirements.txt` | Add `torchaudio` (provides `nn.CTCLoss` + beam-search utilities) |
| `README.md` | New section "Phase 1c: CTC Fingerspelling" |

---

## Implementation Order

1. **Synthetic data generator** + tests (~half day)
   - Concatenate landmark vectors with transitions
   - Save as `data/synthetic_fs/<seq_id>.npz` with `(T, 63)` features and label string
2. **CTC model + training script** (~1 day)
   - Bi-LSTM encoder, linear head, `nn.CTCLoss`
   - Validation: word error rate (WER) on held-out synthetic + real
3. **Greedy decoder** (one function, fast to write)
4. **Beam-search decoder** with optional letter language model (~half day)
5. **`transcribe_video.py`** entry point — read video, extract landmarks, decode
6. **Real-video evaluation** on `example_videos/alphabet.mp4` and others
7. **Optional:** ChicagoFSWild fine-tuning if synthetic underperforms

Estimated effort: ~3 working days for v1, plus 1-2 days of tuning.

---

## Evaluation

**Primary metric:** Letter Error Rate (LER) — Levenshtein distance between
predicted and true letter sequence, normalized by sequence length.

**Targets:**
- Synthetic test set: LER < 5% (sanity check — should be easy)
- `alphabet.mp4` real video: LER < 30% (current best is ~70-85%)
- Stretch: LER < 15% on alphabet.mp4 with beam search + letter LM

**Visualizations:**
- Alignment plot: per-frame letter probabilities over time, overlaid with
  the decoded sequence
- Confusion matrix on letter substitutions
- Failure cases: side-by-side video frame + predicted vs. true alignment

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Synthetic transitions don't match real video | Add Gaussian noise to transitions; fall back to ChicagoFSWild |
| LSTM overfits to synthetic | Heavy augmentation (already designed); held-out real-video val |
| CTC decoder produces blank-only output (collapses) | Standard fixes: smaller LR, gradient clipping, longer warmup |
| Real fingerspelling videos have varying fps | Resample landmark stream to fixed 30fps before model |

---

## Out of Scope

- Word-level signs (covered by [[OPENHANDS_PLAN]])
- Sentence translation (Phase 3)
- Replacing the trained PPCA model — we keep PPCA as an optional
  preprocessing step on the front of the LSTM

---

## References

1. Graves, A. et al. *Connectionist Temporal Classification.* ICML 2006.
2. Shi, B. et al. *Fingerspelling Recognition in the Wild with Iterative Visual Attention.* ICCV 2019. (ChicagoFSWild dataset)
3. PyTorch `nn.CTCLoss` docs.
