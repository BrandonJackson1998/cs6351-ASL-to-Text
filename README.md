# ASL Video-to-Text

**CS-6351: AI & ML — Course Project**
**Team:** Brandon Jackson & Kevin Bateman

A pipeline that takes a fingerspelling video and produces a letter sequence,
using MediaPipe Hands for landmark extraction and a stack of classical and
neural ML models for letter recognition.

See [`docs/STATUS_REPORT.md`](docs/STATUS_REPORT.md) for the full experiment
log and rationale behind the final architecture.

---

## Current state

The current model is a **Random Forest** trained on a class-balanced merge
of Kaggle ASL Alphabet (clean stills) and ChicagoFSWild forced-aligned
frames (real video). Per-frame letter classification reaches **82.7% test
accuracy** on real fingerspelling video frames.

Given a video, the pipeline produces a string of fingerspelled letters.

### Limitations

- **No spaces yet.** Output is a single string with consecutive letter
  holds collapsed. Word boundaries are visible to humans but not segmented.
- **No spell correction.** Noise letters appear between recognized words
  (e.g., `FIGHFIGPDATEY...`).
- **J and Z are weak.** Both are *motion-based* letters (a curl for J, a
  Z-trace for Z). The per-frame classifier sees only static hand shapes,
  so it can't distinguish them reliably.

These are tracked in [Future work](#future-work) below.

---

## Headline results

| Model | Task | Test result |
|---|---|---|
| Mixture-of-PPCAs (Kaggle) | Static image → letter | **98.2% accuracy** |
| **Random Forest (balanced)** | Per-frame letter classification on real video | **82.7% accuracy** |
| **CTC Bi-LSTM v2** | Real fingerspelled video → letter sequence | **47.1% LER on FSWild test** |

On `example_videos/Finger.mp4` (10 fingerspelled fruit names, recorded
fresh), the Random Forest hold decoder finds **all 10 target words** as
recognizable substrings.

---

## Quick start

### Setup

```bash
make install-mac         # macOS
make install             # Linux
```

The MediaPipe Hands model (`data/hand_landmarker.task`, 7.5 MB) downloads
automatically on first use.

### Demo: video → letter sequence

```bash
make transcribe-holds VIDEO=example_videos/Finger.mp4
```

Outputs:

```
FIGHFIGPDATEYDATEQLIMENLIMEQGUAVANGRAUAQOLIVENOLIUEQPRNEPRETN
GONANGQPAPAYANANAYAGUMUATNKRNATNASIONFUITNASIONFRUIT
```

You can read the fruit names through the noise: `FIG`, `DATE`, `LIME`,
`GUAVA`, `OLIVE`, `(P)RUNE`, `(M)ANGO`, `PAPAYA`, `(PA)SSIONFRUIT`,
`FRUIT`.

### Side-by-side model visualization

```bash
make model-comparison-grid VIDEO=example_videos/Finger.mp4
```

Generates `segments/<video>/model_comparison_grid.png` with one frame per
detected hold and predictions from RF / MPPCA / PPCA / CTC for that frame.

---

## Pipeline

```
video
  │
  ▼ MediaPipe Hands per frame  →  T × 63 normalized landmarks
  │
  ├── Random Forest per-frame  ─►  letter holds  ─►  letter sequence
  │   (primary, used by `make transcribe-holds`)
  │
  └── CTC Bi-LSTM v2 + char-LM beam search  ─►  letter sequence
      (alternative, used by `make transcribe-v2-fusion`)
```

The pipeline produces a noisy letter sequence. A planned LLM agent layer
will turn these into clean English; not yet implemented.

---

## Reproducing from scratch

The whole repo is regenerable from a single command (≈ 2 hours on CPU,
≈ 13 GB disk):

```bash
make build-data
```

This runs in dependency order:

1. Download Kaggle ASL Alphabet
2. Extract Kaggle landmarks
3. Download ChicagoFSWild (~13 GB)
4. Extract FSWild landmark sequences
5. Train CTC v2 (needed for forced alignment)
6. Forced-align FSWild → per-letter frame buckets
7. Build the balanced merged training set

After `make build-data` finishes, train per-frame models:

```bash
make train-rf EXP=rf_balanced DATA=data/balanced_letter_frames
make train-mppca EXP=mppca_balanced DATA=data/balanced_letter_frames KS="3 5 8"
make refit-ppca-fswild EXP=ppca_balanced DATA=data/balanced_letter_frames
```

### Required tokens

A Kaggle API token is needed (both datasets are Kaggle-hosted). Either:

```bash
# Recommended (new-style access token)
mkdir -p ~/.kaggle && echo KGAT_your_token > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# Or legacy
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

---

## Models we ship

All trained models live under `experiments/`. The `latest` symlink points at
the Phase 1 PPCA artifact (kept for backward compatibility).

| Path | Model | Trained on | Use |
|---|---|---|---|
| `experiments/rf_balanced/` | **Random Forest** | Balanced Kaggle + FSWild (4k/letter) | Per-frame letter classification (primary) |
| `experiments/mppca_balanced/` | Mixture of PPCAs (5–8 / letter) | Balanced merged | Per-frame, generative ablation |
| `experiments/ppca_balanced/` | PPCA + Mixture-of-PPCAs (Phase 1) | Balanced merged | Per-frame baseline |
| `experiments/latest_ctc_v2/` | **CTC Bi-LSTM v2** (3-layer × 256-hidden) | FSWild + Kaggle, weighted | Sequence model |
| `experiments/svm_fswild/` | RBF-SVM | FSWild only | Ablation |

---

## Inference

### Random Forest hold decoder (recommended)

```bash
make transcribe-holds VIDEO=path/to/video.mp4
```

Optional flags:
- `EMISSION=experiments/mppca_balanced/mixture MPPCA=1` — use MPPCA
- `MIN_HOLD=0.5` — minimum letter-hold duration (seconds)
- `SMOOTH=11` — sliding-window majority size (frames)

### CTC sequence model (fast in-distribution fingerspelling)

```bash
# Pure CTC
make transcribe-v2-fusion VIDEO=path/to/video.mp4 LAM=0

# CTC + Random Forest fusion
make transcribe-v2-fusion VIDEO=path/to/video.mp4 EMISSION=experiments/rf_balanced LAM=0.25
```

### Evaluations

```bash
make evaluate-ppca SPLIT=test            # Phase 1 PPCA on Kaggle test
make evaluate-ctc-v2                      # CTC v2 on FSWild test
make evaluate-fusion-rf RF_DIR=experiments/rf_balanced
```

---

## Project structure

```
cs6351-ASL-to-Text/
├── docs/
│   ├── STATUS_REPORT.md        Full experiment log + rationale
│   ├── PPCA_PLAN.md            Phase 1 plan
│   ├── CTC_PLAN.md             Phase 1c plan
│   ├── OPENHANDS_PLAN.md       Phase 2 plan (deferred — see Future work)
│   └── FINAL_PLAN.md           Consolidation plan
│
├── src/
│   ├── preprocessing/
│   │   ├── landmark_extractor.py       MediaPipe Hands → 63D
│   │   ├── dataset.py                  ASLAlphabetDataset (per-letter splits)
│   │   ├── combined_dataset.py         FSWild + Kaggle for CTC training
│   │   ├── chicagofswild.py            FSWild video → landmark sequences
│   │   ├── fswild_letter_frames.py     CTC forced-align → per-letter frames
│   │   ├── balanced_letter_frames.py   Merge + balance + augment
│   │   ├── temporal_augment.py         Time-warp / dropout / jitter
│   │   └── augment.py                  Landmark jitter / rotate / mirror
│   │
│   └── models/
│       ├── ppca.py                     Tipping-Bishop PPCA
│       ├── ppca_classifier.py          PPCALogisticClassifier + Mixture
│       ├── mppca.py                    Mixture of PPCAs (EM)
│       ├── train_ppca.py
│       ├── train_mppca.py
│       ├── train_rf.py                 ★ Random Forest (the headliner)
│       ├── train_svm.py                RBF-SVM ablation
│       ├── refit_ppca_fswild.py        Refit PPCA on balanced data
│       ├── ctc_lstm.py                 Constants used by decoder
│       ├── ctc_lstm_v2.py              3-layer × 256 Bi-LSTM
│       ├── train_ctc_v2.py
│       ├── ctc_forced_align.py         Constrained Viterbi
│       ├── decode_ctc.py               Greedy + beam + LM-aware beam
│       ├── char_lm.py                  6-gram char language model
│       ├── evaluate.py                 PPCA / MLP eval
│       ├── mlp.py                      Phase 1 MLP baseline
│       └── train_mlp.py
│
├── scripts/
│   ├── transcribe_holds.py             ★ Primary inference (RF hold decoder)
│   ├── transcribe_v2_fusion.py         CTC v2 + optional fusion
│   ├── model_comparison_grid.py        4-model side-by-side viz
│   ├── evaluate_ctc_v2.py              One-shot test eval
│   ├── evaluate_fusion.py              Fusion sweep (PPCA emission)
│   ├── evaluate_fusion_rf.py           Fusion sweep (RF emission)
│   ├── analyze_frame.py                Phase 1 single-frame inference
│   ├── visualize_ppca.py               PPCA visualizations
│   ├── download_data.py                Kaggle ASL Alphabet downloader
│   └── download_fswild.py              ChicagoFSWild downloader
│
├── data/                                Datasets (gitignored, regenerable)
├── experiments/                         Trained model checkpoints (gitignored)
├── segments/                            Per-video pipeline outputs (gitignored)
├── example_videos/                      Demo videos
├── Makefile
└── requirements.txt
```

---

## Future work

These were scoped out for the course project and are good candidates for a
follow-up:

- **Word boundary / spacing detection.** Output is currently one continuous
  string. Hand-drop gaps and longer pauses are visible in the landmark
  stream but the current decoder doesn't use them.
- **Spell correction / dictionary snapping.** Snap noisy substrings to the
  closest English word via edit distance against a vocabulary. Cheap to
  build, would clean up demo output substantially.
- **Motion-based letters (J, Z).** These need a temporal model that sees
  finger trajectories, not single-frame poses. The current RF treats every
  frame independently and can't see the J curl or Z trace.
- **LLM agent layer (Phase 3).** Take noisy letter sequences and produce
  clean English text. Plan in [`docs/CTC_PLAN.md`](docs/CTC_PLAN.md).
- **Word-level sign recognition (Phase 2).** [OpenHands](https://github.com/AI4Bharat/OpenHands)
  is a pretrained PyTorch toolkit with WLASL2000 checkpoints. We had it
  integrated and it works out of the box, but stripped it out to keep the
  repo focused on fingerspelling. Plan: [`docs/OPENHANDS_PLAN.md`](docs/OPENHANDS_PLAN.md).
- **Adaptive thresholds per video.** Hold-duration thresholds were
  hard-coded; a per-video calibration based on the input's typical hold
  length would generalize across signing speeds.

---

## Tech stack

| Component | Tool |
|---|---|
| Hand detection | MediaPipe Hands (21 keypoints × 3 = 63 features) |
| Video processing | OpenCV |
| ML modeling | scikit-learn (RF, SVM, LogReg), PyTorch (CTC LSTM), custom (PPCA, MPPCA) |
| Datasets | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), [ChicagoFSWild](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) |
| Hardware target | CPU (Mac, no GPU) |

---

## References

1. Tipping & Bishop (1999). *Probabilistic Principal Component Analysis.* JRSS-B 61(3).
2. Tipping & Bishop (1999). *Mixtures of Probabilistic Principal Component Analyzers.* Neural Computation 11(2).
3. Graves et al. (2006). *Connectionist Temporal Classification.* ICML.
4. Shi et al. (2018). *American Sign Language fingerspelling recognition in the wild.* IEEE SLT. (ChicagoFSWild)
5. Breiman (2001). *Random Forests.* Machine Learning 45(1).
