# Manual Test Commands

Quick commands to verify the system works.

---

## 1. Quick Full Test (30 seconds)

```bash
./test_all.sh
```

Expected: All tests pass ✓

---

## 2. Individual Component Tests

### Unit Tests
```bash
PYTHONPATH=. .virtual_environment/bin/pytest tests/ -v
```
Expected: 4 tests pass (PPCA tests)

### Hand Crop Extractor
```bash
.virtual_environment/bin/python src/preprocessing/hand_crop_extractor.py
```
Expected: "✓ Hand crop extracted: (200, 200, 3)"

---

## 3. Model Tests

### Check Models Exist
```bash
ls -lh experiments/rf_balanced/model.pkl
ls -d experiments/ppca_*/mixture
```
Expected: RF model (2.2GB) and PPCA directory

### Check Model Accuracy
```bash
cat experiments/rf_balanced/metrics.json | python3 -m json.tool | grep test_acc
cat experiments/ppca_20260616_091014/metrics.json | python3 -m json.tool | grep test_acc
```
Expected: RF ~83%, PPCA ~74%

---

## 4. Transcription Tests

### RF Transcription (if you have test_ensemble.mp4)
```bash
make transcribe-holds VIDEO=test_ensemble.mp4 MIN_HOLD=0.1
```
Expected: "Decoded sequence: NES"

### Ensemble Transcription
```bash
make transcribe-ensemble VIDEO=test_ensemble.mp4 MIN_HOLD=3 RF_WEIGHT=0.9
```
Expected: "TRANSCRIPTION: TNEWS"

---

## 5. Data Verification

### Check Hand Crops
```bash
ls -d data/hand_crops_kaggle/[A-Z] | wc -l
find data/hand_crops_kaggle -name "*.jpg" | wc -l
```
Expected: 19 directories, ~41,000 images

### Check Balanced Dataset
```bash
PYTHONPATH=. .virtual_environment/bin/python -c "
from src.preprocessing.dataset import ASLAlphabetDataset
ds = ASLAlphabetDataset('data/balanced_letter_frames/landmarks', split='test')
print(f'Test samples: {len(ds)}')
print(f'Classes: {ds.num_classes}')
"
```
Expected: 20,800 samples, 26 classes

---

## 6. Ensemble Evaluation

### Compare All Models
```bash
PYTHONPATH=. .virtual_environment/bin/python scripts/evaluate_ensemble.py \
    --ppca-model experiments/ppca_20260616_091014 \
    --rf-weight 0.9
```
Expected: 
- RF: 83.17%
- PPCA: 74.38%
- Ensemble: ~83.3%

---

## 7. Performance Test

### Model Inference Speed
```bash
PYTHONPATH=. .virtual_environment/bin/python -c "
import time
import pickle
import numpy as np

# Load RF
with open('experiments/rf_balanced/model.pkl', 'rb') as f:
    rf = pickle.load(f)

# Time 100 predictions
X = np.random.rand(100, 63).astype(np.float32)
start = time.time()
preds = rf.predict(X)
elapsed = (time.time() - start) / 100 * 1000

print(f'RF inference: {elapsed:.2f}ms per frame')
"
```
Expected: ~1-2ms per frame

---

## 8. Quick Sanity Checks

### Python Environment
```bash
.virtual_environment/bin/python --version
.virtual_environment/bin/pip list | grep -E "scikit-learn|numpy|opencv|mediapipe|pytest"
```
Expected: Python 3.12+, all packages installed

### Git Status
```bash
git status --short | head -20
```
Expected: ~10 files to commit

### Test Video
```bash
[ -f test_ensemble.mp4 ] && echo "✓ Test video exists" || echo "✗ Test video missing"
```

---

## Expected Results Summary

✅ All tests should pass  
✅ RF: 83.17% accuracy  
✅ PPCA: 74.38% accuracy  
✅ Ensemble: 83.32% accuracy  
✅ Hand crops: 19/26 letters, ~41k images  
✅ Inference: <5ms per frame  

---

## Quick One-Liner Test

```bash
./test_all.sh && echo "✅ Everything works!"
```
