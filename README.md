# ASL Video-to-Text

**CS-6351: AI & ML — Course Project**  
**Team:** Brandon Jackson & Kevin Bateman

Pipeline for converting ASL video to text using MediaPipe Hands for landmark extraction and neural models for recognition.

**Capabilities:**
- Fingerspelling: 97.5% accuracy (26 letters)
- Word segmentation: Automatic boundary detection
- Word recognition: Temporal CNN (currently 4 words, expandable to 100s with large datasets)

See [`HANDOFF.md`](HANDOFF.md) for project setup and training instructions.  
See [`docs/STATUS_REPORT.md`](docs/STATUS_REPORT.md) for full experiment history.

---

## Current state

The pipeline offers two per-frame classifiers:
- **Random Forest (landmarks)**: 83.2% accuracy, fast, works on any device
- **EfficientNet-B0 (hand crops)**: 97.5% accuracy, CNN-based

Both are trained on a class-balanced merge of Kaggle ASL Alphabet (clean 
stills) and ChicagoFSWild forced-aligned frames (real video).

Given a video, the pipeline produces a string of fingerspelled letters.

### Limitations

- **Noisy output.** Filler letters appear between words during hand transitions.
- **J and Z are weak.** Both are *motion-based* letters requiring temporal 
  models to see finger trajectories.

---

## Headline results

| Model | Task | Test result |
|---|---|---|
| Mixture-of-PPCAs (Kaggle) | Static image → letter | **98.2% accuracy** |
| **Random Forest (balanced)** | Per-frame letter (landmarks) | **83.2% accuracy** |
| **EfficientNet-B0** | Per-frame letter (hand crops) | **97.5% accuracy** |
| **CTC Bi-LSTM v2** | Real fingerspelled video → letter sequence | **47.1% LER on FSWild test** |

On test videos with 10 fingerspelled fruit names, the Random Forest hold 
decoder finds **all 10 target words** as recognizable substrings.

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

**Note:** Models and example videos are not included in the repo. Train models with `make build-data` first.

**Letter-level output (no word boundaries):**
```bash
make transcribe-holds VIDEO=path/to/video.mp4
```

Example output from a fruit-naming video:
```
FIGHFIGPDATEYDATEQLIMENLIMEQGUAVANGRAUAQOLIVENOLIUEQPRNEPRETN
```

**Word-level output (with segmentation):**
```bash
make transcribe-words VIDEO=path/to/video.mp4
```

Example output with word boundaries:
```
FIG FIG | DATE DATE | LIME LIME | GUAVA GUAVA | OLIVE OLIVE
```

Word segmentation detects boundaries via hand-drop gaps and pause duration.
See [`docs/WORD_SEGMENTATION.md`](docs/WORD_SEGMENTATION.md) for details.

---

## Pipeline

```
video
  │
  ▼ MediaPipe Hands per frame  →  T × 63 normalized landmarks
  │
  ├── Random Forest per-frame  ─►  letter holds  ─►  noisy letter sequence
  │   (primary, used by `make transcribe-holds`)
  │
  └── CTC Bi-LSTM v2 + char-LM beam search  ─►  noisy letter sequence
      (alternative, used by `make transcribe-v2-fusion`)
```

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

## Models

**Note:** Trained models are not included in the repo (too large for git). 
Train them yourself using `make build-data` followed by training commands above.

Key models:
- **EfficientNet-B0** (efficientnet_b0) - Per-frame classification, 97.5% accuracy
- **Random Forest** (rf_balanced) - Per-frame classification, 83.2% accuracy
- **CTC Bi-LSTM v2** - Sequence model, 47.1% letter error rate
- **PPCA/MPPCA** - Baselines, 98.2% on static images

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
src/
├── preprocessing/          Feature extraction & data pipelines
└── models/                 Model training (RF, CTC, PPCA, EfficientNet)

scripts/
├── transcribe_holds.py    Primary inference script
├── extract_hand_crops.py  Hand crop extraction for CNN training
└── download_*.py          Data downloaders

docs/                      Experiment logs and plans
data/                      Datasets (regenerable, gitignored)
experiments/               Model checkpoints (gitignored)
tests/                     Test suite
```

---

## Future work

- **LLM-based cleanup** (Phase 3). Pass segmented output to Claude API to
  correct noisy letter sequences and produce grammatical English text.
- **Motion-based letters (J, Z).** These need a temporal model that sees
  finger trajectories, not single-frame poses. Both classifiers treat frames
  independently and can't see the J curl or Z trace.
- **Word-level sign recognition.** [OpenHands](https://github.com/AI4Bharat/OpenHands)
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
