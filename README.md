# ASL Video-to-Text Translation

**CS-6351: AI & ML -- Course Project**  
**Team:** Brandon Jackson & Kevin Bateman

## Project Status

**Phase 1 Complete:** Baseline fingerspelling classifier trained and tested.

- ✅ Full pipeline implemented (download → train → evaluate → inference)
- ✅ MLP model trained on [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87k static images)
- ⚠️ **Current accuracy: 7.7%** on [test video](https://www.youtube.com/shorts/Fd29yKrZttE) (A-Z fingerspelling)
- 🔧 **Known issue:** Training/test mismatch (static images vs. dynamic video)

**Next:** Debug segmentation, add data augmentation, improve keyframe selection.

---

## Prerequisites

Before running the pipeline, ensure you have:

### 1. Python 3.12+
```bash
python3 --version  # Should be 3.12 or higher
```

### 2. Kaggle Authentication (Required for Training)

The pipeline downloads training data from Kaggle. You need a free Kaggle account and authentication credentials.

**Setup (choose ONE method):**

#### Method 1: Access Token (Recommended - New)

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API section
3. Click "Create New Token"
4. Copy and run the command Kaggle provides:

```bash
mkdir -p ~/.kaggle && echo KGAT_your_token_here > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
```

**Verify:**
```bash
[ -f ~/.kaggle/access_token ] && echo "✓ Access token ready" || echo "✗ Token missing"
```

#### Method 2: Legacy API Key (kaggle.json)

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API section
2. Click "Create New API Token" (downloads `kaggle.json`)
3. Install the file:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Verify:**
```bash
[ -f ~/.kaggle/kaggle.json ] && echo "✓ API key ready" || echo "✗ Key missing"
```

**Note:** The pipeline supports both methods. Use whichever Kaggle provides you.

### 3. System Dependencies

**macOS:**
```bash
brew install python@3.12 ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ffmpeg
```

---

## Quick Start

### Setup

```bash
# macOS (~2-3 minutes)
make install-mac

# Linux (~2-3 minutes)
make install
```

This creates a virtual environment and installs Python dependencies (PyTorch, OpenCV, MediaPipe, etc.).

**Note:** The MediaPipe hand detection model (`data/hand_landmarker.task`, 7.5MB) will be automatically downloaded on first use.

### Option A: Automated Pipeline (Recommended)

**Prerequisites:** Kaggle authentication required (see Prerequisites section above)

```bash
./run_full_pipeline.sh                # Uses cached data when available
./run_full_pipeline.sh --clear-cache  # Fresh start
```

**Time:** ~30-40 min (CPU) first run, seconds if cached.

---

### Option B: Manual Step-by-Step

#### 1. Train the Fingerspelling Classifier

```bash
make download-alphabet       # ~5-10 min (1GB download, requires Kaggle API token)
make extract-landmarks       # ~15-20 min (87k images → hand landmarks, cached)
make train-baseline          # ~10-15 min (20 epochs, CPU) / ~2-3 min (GPU)
make evaluate-baseline       # ~1-2 min
```

**Prerequisites:** Kaggle API token at `~/.kaggle/kaggle.json` (get from [kaggle.com/settings](https://www.kaggle.com/settings))

#### 2. Analyze ASL Video

```bash
make string-deltas VIDEO=https://www.youtube.com/shorts/Fd29yKrZttE CONFIDENCE=0.7 VERBOSE=1
```

**Time:** ~1-2 min first run, ~1-2 sec cached. Add `CONFIDENCE=0.7` to filter low-confidence predictions.

---

### Option C: Clear Caches

```bash
# Clear everything (re-run full pipeline)
rm -rf data/asl_alphabet/ experiments/ segments/

# Clear only trained models (retrain from cached landmarks)
rm -rf experiments/

# Clear only video analyses (reprocess videos)
rm -rf segments/

# Clear specific video
rm -rf segments/<video-id>/
```

---

### Check Current State

```bash
# View what's cached
ls -lh data/asl_alphabet/           # Training data
ls -lh experiments/latest/          # Trained model
ls -lh segments/                    # Video analyses

# Verify pipeline readiness
[ -f experiments/latest/best_model.pt ] && echo "✓ Model ready" || echo "✗ Need to train"
```

---

## Pipeline Architecture

```
Video → Segmentation → Keyframe Extraction → Hand Landmarks → MLP Classifier → Letter Sequence
         (cached)        (cached)              (per-frame)      (trained)        (deduplicated)
```

**Frame-rate independence:** All temporal parameters use seconds (e.g., `min_segment_duration=0.5s`, `min_spacing_seconds=0.4s`), converted to frames via fps. Works correctly on slow-motion (60-120fps), real-time (30fps), or time-lapse (15fps) videos.

### Caching Strategy

- **Video segmentation**: Clips cached to `segments/<video-id>/clips/`. First run: ~10-30s. Cached: instant.
- **Keyframe extraction**: Images cached to `segments/<video-id>/keyframes/`. First run: ~10-30s. Cached: instant.
- **Landmark extraction (training)**: Features cached to `data/asl_alphabet/landmarks/`. First run: ~15-20 min (87k images). Cached: instant.
- **Model checkpoints**: Saved to `experiments/<timestamp>/best_model.pt`. Symlinked as `experiments/latest/`. Training: ~10-15 min (CPU) / ~2-3 min (GPU).

---

## Current Status & Performance

**Model:** 3-layer MLP trained on ASL Alphabet dataset (87k static images, 29 classes)  
**Checkpoint:** `experiments/latest/best_model.pt` (246KB)

### Test Results

**Test video:** A-Z fingerspelling ([youtube.com/shorts/Fd29yKrZttE](https://www.youtube.com/shorts/Fd29yKrZttE))

```bash
make string-deltas VIDEO=https://www.youtube.com/shorts/Fd29yKrZttE CONFIDENCE=0.7
```

**Results:**
- **Positional accuracy: 7.7%** (2/26 letters in correct position)
- **Letter recall: 65.4%** (17/26 letters detected somewhere in sequence)
- **Model confidence: 70-100%** (high confidence, but mostly incorrect)

**Key Issue:** Model overconfident on wrong predictions. Trained on static frontal images, tested on dynamic video with motion.

### Known Problems

1. **Segmentation/Keyframe Selection** - Capturing transition frames instead of letter holds
2. **Training/Test Mismatch** - Static training images vs. dynamic video with motion blur
3. **Missing Letters** - Cannot detect: A, G, H, J, M, P, Q, R, U
4. **Over-segmentation** - 31 predictions for 26 letters (duplicates: C, E, N, S, X, Y, Z)

### Next Steps

**Priority fixes:**
1. Visualize keyframes to debug segmentation (`segments/Fd29yKrZttE/keyframes/`)
2. Add data augmentation (rotation, brightness, motion blur)
3. Improve keyframe selection (better velocity thresholds)
4. Train on video frames, not just static images

**Future improvements:**
- Temporal models (LSTM/Transformer) for sequence context
- Multi-frame aggregation instead of single-frame classification
- Calibrate confidence scores (currently miscalibrated)

---

## Datasets

**Training:** [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) - 87k static images, 29 classes (A-Z + SPACE, DELETE, NOTHING)  
**Test video:** [youtube.com/shorts/Fd29yKrZttE](https://www.youtube.com/shorts/Fd29yKrZttE) - A-Z fingerspelling demo  
**Future (Phase 2):** [WLASL](https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized) - 2k word-level video clips

---

## Project Structure

```
cs6351-ASL-to-Text/
├── run_full_pipeline.sh            # Automated full pipeline script
├── Makefile                        # CLI commands
├── requirements.txt                # Python dependencies
├── src/
│   ├── preprocessing/
│   │   ├── video_segmenter.py      # Video → sign clips (wrist velocity heuristics)
│   │   ├── keyframe_extractor.py   # Clips → representative keyframe images
│   │   ├── landmark_extractor.py   # MediaPipe hand landmark extraction
│   │   └── dataset.py              # PyTorch dataset loaders
│   ├── models/
│   │   ├── mlp.py                  # MLP fingerspelling classifier
│   │   ├── train_mlp.py            # Training script
│   │   └── evaluate.py             # Evaluation metrics
│   └── agents/                     # (stub) Future LLM translation layer
├── scripts/
│   ├── download_data.py            # Kaggle dataset downloader
│   ├── analyze_frame.py            # Single-frame inference
│   ├── string_deltas.py            # End-to-end video → letter sequence
│   └── visualize_segmentation.py   # Segmentation debug plots
├── data/                           # Datasets (gitignored)
│   ├── asl_alphabet/               # ASL Alphabet images + landmarks
│   └── hand_landmarker.task        # MediaPipe model file
├── experiments/                    # Model checkpoints (gitignored)
│   └── latest/                     # Symlink to most recent training run
├── segments/                       # Video analysis outputs (gitignored)
│   └── <video-id>/
│       ├── clips/                  # Segmented video clips
│       ├── keyframes/              # Extracted keyframe images
│       └── predictions.json        # Model predictions + deduped sequence
├── Makefile                        # CLI commands
└── requirements.txt                # Python dependencies
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Hand Detection | MediaPipe Hands (21-point landmarks) |
| Video Processing | OpenCV |
| Model Training | PyTorch |
| Dataset | ASL Alphabet (Kaggle) |

---

## Makefile Reference

**Setup:**
- `make install` / `make install-mac` — Install dependencies (~2-3 min)

**Training:**
- `make download-alphabet` — Download training data (~5-10 min, 1GB)
- `make extract-landmarks` — Extract hand landmarks, cached (~15-20 min first run, instant thereafter)
- `make train-baseline` — Train MLP classifier (~10-15 min CPU / ~2-3 min GPU)
- `make evaluate-baseline` — Evaluate model (~1-2 min)

**Inference:**
- `make string-deltas VIDEO=<path>` — Full pipeline, video → letters (~1-2 min first run, ~1-2 sec cached)
- `make analyze-frame FRAME=<path>` — Single-frame prediction (<1 sec)
- `make segment-video VIDEO=<path>` — Segment video only (~10-30 sec)
- `make extract-keyframes VIDEO=<path>` — Extract keyframes only (~10-30 sec)

**Flags:**
- `VERBOSE=1` — Show per-frame predictions
- `CONFIDENCE=<float>` — Filter predictions below threshold (default: 0.0)
- `CHECKPOINT=<path>` — Use specific model checkpoint

---

## Team Responsibilities

| Area | Owner |
|------|-------|
| Video segmentation, keyframe extraction, preprocessing pipeline | Brandon Jackson |
| Model training, inference scripts, evaluation | Kevin Bateman |
| Integration, testing, documentation | Both |

---

## References

1. Abeyta et al. *ASL Alphabet Dataset*. Kaggle, 2018.
2. Google. *MediaPipe Hands*.
