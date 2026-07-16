# ASL Video-to-Text: Handoff to Brandon

**Date:** 2026-07-13  
**Status:** Ready for large-dataset training

---

## Current Capabilities

1. **Fingerspelling:** 97.5% accuracy on 26 letters (EfficientNet-B0)
2. **Word Segmentation:** Automatic boundary detection via hand-drop/pause
3. **Word Recognition:** 4 words trained (YOU, DO, WANT, I) - inference working
4. **Pipeline:** YouTube URL → landmarks → trained model → predictions

---

## Project Structure

```
src/
├── models/
│   ├── word_sign_recognizer.py      # Temporal CNN architectures
│   ├── train_word_sign.py           # Training pipeline
│   └── train_efficientnet.py        # Fingerspelling CNN
└── preprocessing/
    └── landmark_extractor.py         # MediaPipe extraction

scripts/
├── download_youtube_clip.py          # Video downloader
├── extract_wlasl_landmarks.py        # Landmark extraction
├── organize_individual_words.py      # Dataset organization
├── recognize_word.py                 # Word inference
└── transcribe_with_segmentation.py   # Segmented transcription

docs/
├── STATUS_REPORT.md                  # Full experiment history (June)
├── IMPROVEMENT_PLAN.md               # EfficientNet results
└── WORD_SEGMENTATION.md              # Segmentation details
```

---

## Quick Start

### Run Inference

**Fingerspelling:**
```bash
make transcribe-holds VIDEO=video.mp4
```

**Word segmentation:**
```bash
make transcribe-words VIDEO=video.mp4
```

**Word recognition:**
```bash
python scripts/recognize_word.py \
    --input clip.mp4 \
    --model experiments/expanded_words_v1/best_model.pth \
    --vocabulary data/expanded_word_dataset/vocabulary.json
```

### Train New Model

```bash
# 1. Organize your dataset
python scripts/organize_individual_words.py \
    --landmarks-dir data/your_landmarks \
    --output-dir data/your_dataset \
    --min-samples 2

# 2. Train
python -m src.models.train_word_sign \
    --data-dir data/your_dataset \
    --model-type cnn \
    --experiment your_experiment \
    --epochs 200
```

---

## Current Models

Located in `experiments/` (gitignored):

1. **rf_balanced/** - Random Forest fingerspelling (83%)
2. **efficientnet_b0/** - CNN fingerspelling (97.5%)
3. **expanded_words_v1/** - 4-word temporal CNN (100% val, working inference)

---

## For Large Dataset Training

### Current Architecture Limitations
- **SimpleTemporalCNN:** 750K params, designed for small datasets
- **Limited to landmarks:** 63D MediaPipe features only
- **No pretraining:** Training from scratch

### Recommendations for 50GB/20GB Datasets

**Option A: Use Pretrained Model (Recommended)**
- OpenHands ST-GCN (pretrained on WLASL)
- Or I3D/Slowfast (video-based)
- Fine-tune on your datasets
- Expected: 60-80% accuracy on 100-2000 words

**Option B: Scale Current Architecture**
- Use TemporalCNN_LSTM (3.5M params) instead of SimpleTemporalCNN
- Add data augmentation (time warping, noise)
- Increase batch size and learning rate with GPU
- Expected: 40-60% accuracy on 100-500 words

### Setup for University Cluster

See `requirements.txt` for dependencies. Key packages:
- PyTorch 2.11.0
- MediaPipe 0.10.14
- timm, albumentations (for EfficientNet)

---

## Data Requirements Learned

From small-scale experiments:
- **1 sample/word:** Unstable (78% train, high variance)
- **2-5 samples/word:** Overfits (100% val, 0% test)
- **10-20 samples/word:** Good generalization (70-80% test expected)
- **50+ samples/word:** Excellent (85%+ test expected)

Your large datasets should eliminate data scarcity issues.

---

## Key Files to Know

**Training:**
- `src/models/train_word_sign.py` - Main training script
- `src/models/word_sign_recognizer.py` - Model definitions

**Data Processing:**
- `scripts/extract_wlasl_landmarks.py` - Extract MediaPipe landmarks
- `src/preprocessing/landmark_extractor.py` - Core extraction logic

**Inference:**
- `scripts/recognize_word.py` - Single clip recognition
- `scripts/transcribe_with_segmentation.py` - Full video pipeline

**Documentation:**
- `README.md` - Overview and results
- `docs/STATUS_REPORT.md` - Complete experiment log
- `MANUAL_TEST_COMMANDS.md` - Example commands

---

## What's Gitignored

- `data/` - All datasets (too large)
- `experiments/` - Trained models (regenerable)
- `*.mp4` - Videos (regenerable from YouTube)
- `.virtual_environment/` - Python venv

**To reproduce:** Run `make build-data` (for fingerspelling data) or retrain models.

---

## Tests

Run test suite:
```bash
python -m pytest tests/
```

Key tests:
- `tests/test_word_sign_pipeline.py` - 7 pipeline tests
- `tests/test_segmentation_logic.py` - 4 segmentation tests

All should pass.

---

## Next Steps for Brandon

1. **Identify your datasets:** What are the 50GB and 20GB datasets?
2. **Choose architecture:** Pretrained (OpenHands) or scaled current model?
3. **Set up cluster:** Upload data, install dependencies
4. **Train:** Fine-tune on large datasets (days with GPUs)
5. **Integrate:** Replace `experiments/expanded_words_v1/` with new model
6. **Evaluate:** Test on held-out data

---

## Contact

Questions about the codebase? Check:
- `README.md` - High-level overview
- `docs/STATUS_REPORT.md` - Full experiment history
- Code comments in key files

Everything is ready for large-scale training.
