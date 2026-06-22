# Quick Start Guide

Get up and running with ASL-to-Text in 5 minutes.

## Prerequisites

```bash
# macOS
brew install python@3.12 ffmpeg
# or Ubuntu
sudo apt-get install python3.12 python3.12-venv ffmpeg
```

## Setup (2 minutes)

```bash
# 1. Clone and enter directory
cd cs6351-ASL-to-Text

# 2. Create virtual environment and install dependencies
make install-mac  # or 'make install' for Linux

# 3. Verify installation
./test_all.sh
```

## Test the System (3 minutes)

### Option 1: Use Existing Test Video

```bash
# RF transcription (83% accuracy)
make transcribe-holds VIDEO=test_ensemble.mp4 MIN_HOLD=0.1

# Output: NES
```

### Option 2: Transcribe Your Own Video

```bash
# Record yourself fingerspelling or use any ASL video
make transcribe-holds VIDEO=your_video.mp4
```

## Key Commands

| Task | Command |
|------|---------|
| **Transcribe video** | `make transcribe-holds VIDEO=file.mp4` |
| **Test everything** | `./test_all.sh` |
| **Run unit tests** | `make test` |
| **Train RF model** | `make train-rf EXP=name DATA=path` |
| **Train EfficientNet** | `make train-efficientnet EXP=name` |
| **Extract hand crops** | `make extract-hand-crops` |

## What's Working

✅ **EfficientNet** - 97.5% accuracy (CNN on hand crops)  
✅ **RF Transcription** - 83.2% accuracy (landmarks)  
✅ **Ensemble** - 83.3% accuracy (RF + PPCA)  
✅ **Hand Crop Extraction** - 60k images extracted @ 2300/sec  
✅ **4 Unit Tests** - All passing

## Next Steps

- **Use it:** Transcribe ASL videos (83% RF or 97.5% EfficientNet)
- **Integrate EfficientNet:** Wire up EfficientNet to transcription pipeline
- **Extend it:** Add word segmentation, J/Z motion detection

## Documentation

- **README.md** - Project overview and detailed setup
- **docs/IMPROVEMENT_PLAN.md** - Roadmap to 95% accuracy with EfficientNet

## Getting Help

```bash
# Run system test
./test_all.sh

# Check model accuracy
cat experiments/rf_balanced/metrics.json

# List all make targets
grep "^[a-z]" Makefile | cut -d: -f1
```

## Troubleshooting

**Issue:** `make transcribe-holds` fails  
**Fix:** Ensure models exist: `ls experiments/rf_balanced/model.pkl`

**Issue:** No models found  
**Fix:** Models should already be trained. Check `experiments/` directory.

**Issue:** Test video missing  
**Fix:** Use any ASL video or create one from images

---

**Ready!** You now have a working ASL fingerspelling recognition system.
