# Model Accuracy Improvement Plan

**Goal:** Increase per-frame accuracy from 83% to 95%+  
**Approach:** Add EfficientNet transfer learning on hand crop images  
**Inspiration:** [Kaggle ASL F1 Score 99.96%](https://www.kaggle.com/code/gpiosenka/asl-f1-score-99-96)

---

## Current State

### What We Have
- **Random Forest classifier:** 83% accuracy on real video
- **Input:** 63D MediaPipe landmarks (21 keypoints × 3 coordinates)
- **Inference speed:** ~1ms per frame (very fast)
- **Training data:** 104,000 balanced frames (4,000 per letter)

### The Limitation
**Information loss:** MediaPipe landmarks capture only hand skeleton (63 numbers) but discard:
- Finger spacing and angles
- Hand texture and shadows
- Subtle pose variations
- Exact finger positions

**Result:** Maximum possible accuracy is bounded by information in 63D features.

---

## Proposed Solution: Transfer Learning with EfficientNet

### Why This Works

**Kaggle approach achieved 99.96% accuracy using:**
1. Raw pixel data (200×200×3 = 120,000 values vs. our 63)
2. EfficientNet CNN (pretrained on ImageNet)
3. Aggressive regularization (dropout=0.4, L1/L2)
4. Transfer learning (leverages millions of pretrained images)

**Our advantage:** We already extract hand landmarks, so cropping to hand region is trivial.

---

## Implementation Plan

### Phase 1: Data Pipeline Modification (4-6 hours)

**Goal:** Generate hand crop images from video frames

#### Step 1.1: Create Hand Crop Extractor
**File:** `src/preprocessing/hand_crop_extractor.py`

**Functionality:**
- Read video frame + MediaPipe landmarks
- Calculate hand bounding box from landmarks
- Add 20% padding around hand
- Crop frame to bounding box
- Resize to 200×200
- Save as JPG/PNG

**Input:** Video + landmark pickle files  
**Output:** `data/hand_crops/train/A/img_00001.jpg`, etc.

#### Step 1.2: Process Existing Data
**Script:** `scripts/extract_hand_crops.py`

```bash
# Process Kaggle alphabet
python scripts/extract_hand_crops.py \
    --dataset kaggle \
    --output data/hand_crops_kaggle

# Process balanced frames
python scripts/extract_hand_crops.py \
    --dataset balanced \
    --input data/balanced_letter_frames \
    --output data/hand_crops_balanced
```

**Output:** 104,000 hand crop images organized by letter

#### Step 1.3: Update Makefile
Add target:
```makefile
extract-hand-crops:
    python scripts/extract_hand_crops.py \
        --dataset balanced \
        --input data/balanced_letter_frames \
        --output data/hand_crops_balanced
```

---

### Phase 2: EfficientNet Training (6-8 hours)

**Goal:** Train EfficientNet classifier on hand crops

#### Step 2.1: Install Dependencies
```bash
pip install timm  # PyTorch Image Models
pip install torchvision
pip install albumentations  # For augmentation
```

**Update:** `requirements.txt`

#### Step 2.2: Create Training Script
**File:** `src/models/train_efficientnet.py`

**Architecture (from Kaggle):**
```python
import timm
import torch.nn as nn

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super().__init__()
        # Use EfficientNet-B0 (lighter than B3, still effective)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='max'  # Max pooling like Kaggle
        )
        
        # Custom head (matching Kaggle's recipe)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1280, momentum=0.99, eps=0.001),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
```

**Training Configuration (from Kaggle):**
- **Optimizer:** Adamax (lr=0.005)
- **Loss:** CrossEntropyLoss
- **Batch size:** 32
- **Epochs:** 40 with early stopping
- **Regularization:** 
  - L2 weight decay: 0.016
  - L1 activity regularization: 0.006
  - Dropout: 0.4
- **Augmentation (training only):**
  - Horizontal flip
  - Rotation: ±20°
  - Width/height shift: ±20%
  - Zoom: ±20%

**Data Split:**
- Train: 62,400 (60%)
- Val: 20,800 (20%)
- Test: 20,800 (20%)

#### Step 2.3: Training Command
```bash
make train-efficientnet \
    EXP=efficientnet_b0_balanced \
    DATA=data/hand_crops_balanced \
    EPOCHS=40 \
    LR=0.005
```

**Expected training time:**
- CPU: 2-3 hours
- GPU (M1/M2 Mac): 30-45 minutes
- CUDA GPU: 20-30 minutes

**Expected accuracy:** 94-96% on test set

---

### Phase 3: Integration (3-4 hours)

**Goal:** Integrate EfficientNet into inference pipeline

#### Step 3.1: Create Inference Module
**File:** `src/models/efficientnet_inference.py`

**Functionality:**
- Load trained model
- Extract hand crop from frame + landmarks
- Preprocess (resize, normalize)
- Run inference
- Return class probabilities

#### Step 3.2: Add to Transcription Script
**File:** `scripts/transcribe_holds_efficientnet.py`

**Pipeline:**
```
Video frame
  ↓
MediaPipe (extract landmarks + hand bbox)
  ↓
Crop to hand region (200×200)
  ↓
EfficientNet (classify letter)
  ↓
Hold decoder (collapse consecutive letters)
  ↓
Letter sequence
```

#### Step 3.3: Update Makefile
```makefile
transcribe-efficientnet:
    python scripts/transcribe_holds_efficientnet.py \
        --input $(VIDEO) \
        --model experiments/efficientnet_b0_balanced/model.pth \
        --output segments/$(VIDEO)/transcription.txt
```

---

### Phase 4: Evaluation & Comparison (2 hours)

#### Step 4.1: Comparative Evaluation
**Script:** `scripts/compare_models.py`

Run all models on same test set:
- Random Forest (baseline: 83%)
- PPCA (baseline: 98.2% on static)
- EfficientNet (expected: 94-96%)

**Metrics:**
- Overall accuracy
- Per-letter accuracy
- Confusion matrix
- Inference time per frame

#### Step 4.2: Real-World Test
Test on actual fingerspelling video:
```bash
# Record test video
# Run both models
make transcribe-holds VIDEO=test.mp4  # RF
make transcribe-efficientnet VIDEO=test.mp4  # EfficientNet

# Compare outputs
```

---

## Expected Results

### Accuracy Improvement
| Model | Accuracy | Inference Time | Training Time |
|-------|----------|----------------|---------------|
| **Random Forest (current)** | 83% | ~1ms | ~10 min |
| **EfficientNet (proposed)** | 94-96% | ~50ms (CPU) | ~2 hours (CPU) |
| | | ~5ms (GPU) | ~30 min (GPU) |

**Accuracy gain:** +11-13 percentage points

### Per-Letter Improvements
Expected biggest gains on currently challenging letters:
- **H:** 62% → 88% (+26%)
- **I:** 65% → 89% (+24%)
- **K:** 69% → 87% (+18%)
- **O:** 65% → 86% (+21%)
- **R:** 69% → 88% (+19%)

Letters already strong (>95%) should stay near perfect.

---

## Alternative: Quick Ensemble (Optional Quick Win)

**Before implementing EfficientNet, try this 30-minute experiment:**

### Ensemble RF + PPCA
- You already have both trained
- Simple voting: use prediction both models agree on
- When disagree: use model with higher confidence

**Expected result:** 85-87% (2-4% gain with zero work)

**Implementation:**
```python
# scripts/transcribe_ensemble.py
rf_pred = rf_model.predict_proba(features)
ppca_pred = ppca_model.predict_proba(features)

# Weighted average
ensemble = 0.6 * rf_pred + 0.4 * ppca_pred
final_pred = ensemble.argmax()
```

This could be done TODAY while planning EfficientNet implementation.

---

## Timeline Summary

| Phase | Time | Cumulative |
|-------|------|------------|
| **Phase 1:** Data pipeline | 4-6 hours | 6 hours |
| **Phase 2:** Training | 6-8 hours | 14 hours |
| **Phase 3:** Integration | 3-4 hours | 18 hours |
| **Phase 4:** Evaluation | 2 hours | 20 hours |

**Total:** ~2.5 days of development
**Result:** 83% → 95% accuracy (+12% absolute gain)

---

## Success Criteria

### Minimum Success
- ✅ EfficientNet trains without errors
- ✅ Test accuracy ≥ 90%
- ✅ Integration into pipeline works
- ✅ Inference completes on real video

### Target Success
- ✅ Test accuracy ≥ 94%
- ✅ Outperforms RF on all letter categories
- ✅ Reduces confusion on H, I, K, O, R by >15%
- ✅ Maintains or improves on easy letters (Q, V, W, X, Y, Z)

### Stretch Goal
- ✅ Test accuracy ≥ 96%
- ✅ Inference time <10ms per frame on M1 Mac
- ✅ Successfully processes Finger.mp4 with >95% accuracy

---

## Risks & Mitigations

### Risk 1: Slower Inference
**Impact:** 50x slower than RF (50ms vs 1ms)
**Mitigation:** 
- Batch processing (process multiple frames at once)
- Use GPU if available
- Cache predictions for consecutive identical frames
- Consider EfficientNet-Lite for edge devices

### Risk 2: Overfitting
**Impact:** High test accuracy but poor real-world performance
**Mitigation:**
- Use Kaggle's proven regularization
- Test on completely held-out videos
- Use aggressive augmentation

### Risk 3: Hand Crop Quality
**Impact:** Poor crops → poor classification
**Mitigation:**
- Add padding around bounding box (20%)
- Handle edge cases (hand at frame edge)
- Fallback to full frame if crop fails

### Risk 4: Training Time
**Impact:** 2-3 hours on CPU
**Mitigation:**
- Use transfer learning (already 90% there)
- Train on GPU if available
- Start with smaller model (B0 vs B3)

---

## Next Steps

1. **Decide:** Review this plan with Brandon (ensure no conflicts)
2. **Quick win:** Try RF+PPCA ensemble today (30 min)
3. **Commit:** If ensemble shows promise, proceed with EfficientNet
4. **Execute:** Follow phases 1-4 over next 2-3 days
5. **Validate:** Test on real videos, measure accuracy improvement
6. **Document:** Update README with new results

---

## References

- **Kaggle Notebook:** https://www.kaggle.com/code/gpiosenka/asl-f1-score-99-96
- **EfficientNet Paper:** Tan & Le (2019), "EfficientNet: Rethinking Model Scaling for CNNs"
- **TIMM Library:** https://github.com/huggingface/pytorch-image-models
- **Original RF Results:** 82.7% documented in `docs/STATUS_REPORT.md`
