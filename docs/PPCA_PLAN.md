# PPCA-Based ASL Fingerspelling Pipeline — Plan

**Status:** Draft (2026-05-20)
**Owner:** Brandon Jackson & Kevin Bateman
**Replaces:** MLP baseline (Phase 1) — current accuracy 7.7% on test video

---

## Motivation

The Phase 1 MLP baseline reaches good accuracy on held-out static images but
collapses to ~7.7% positional accuracy on real ASL fingerspelling video. Two
hypotheses for the gap:

1. **Train/test mismatch.** Training data is static, well-lit, frontal images.
   Inference frames have motion blur, varied lighting, and transitional poses.
2. **No noise model.** The MLP treats 63D MediaPipe landmarks as exact. In
   practice landmarks are noisy, especially mid-motion.

PPCA (Tipping & Bishop, 1999) directly addresses (2): it models data as a
low-dimensional latent subspace plus isotropic Gaussian noise. Reducing 63D
landmarks to ~12D principal components both **denoises** the input and gives
us a **generative** model we can analyze and visualize for the report.

Augmentation addresses (1).

---

## Architecture

```
Image
  │
  ▼
MediaPipe Hands  ──►  63D landmarks (21 keypoints × xyz)
  │
  ▼
Normalize (wrist-centered, scale-invariant)
  │
  ▼
[Augment: jitter, rotate, mirror]   (train only)
  │
  ▼
PPCA  (63D → ~12D latent)
  │
  ▼
Logistic Regression  (26 classes A–Z)
  │
  ▼
Letter prediction + confidence
```

**Stretch comparison:** Mixture of PPCAs — one PPCA per class, classify by max
log-likelihood under the per-class generative model. Lets us report a "purely
generative" number alongside the discriminative one.

---

## Data Splits

**Source:** Kaggle ASL Alphabet (87k images, 29 classes). We use only A–Z (26
classes). `SPACE`, `DELETE`, `NOTHING` are dropped — the model never needs to
predict them at inference time.

**Split:** **60% train / 20% val / 20% test**, stratified by class, fixed
seed = 42.

| Split | Approx. samples (per class) | Use |
|-------|-----------------------------|-----|
| Train | ~1800 | Fit PPCA, fit classifier, augment |
| Val   | ~600  | Component sweep, hyperparameter selection |
| Test  | ~600  | Final reported accuracy — touched once |

The test split is set aside before any tuning. No model selection decisions
look at the test set.

---

## Augmentation

To narrow the static-vs-video distribution gap, we augment **landmark
features** at training time. Cheap, no re-extraction needed.

**Per-sample, applied with probability ~0.5 each:**

| Augmentation | Range | Why |
|--------------|-------|-----|
| Per-coord Gaussian jitter | σ = 0.02 (post-normalization units) | Simulates landmark detection noise |
| 3D rotation around wrist | ±15° in each axis | Camera angle / hand tilt variation |
| Mirror x-axis | flip | Left-vs-right hand invariance |

**Out of scope for v1:** image-level augmentation (rotation, brightness, motion
blur) before MediaPipe extraction. Adds re-extraction cost (~15-20 min) and
landmark-level should give most of the win.

---

## File-by-File Changes

### New files

| File | Purpose |
|------|---------|
| `src/models/ppca.py` | PPCA class (fit, transform, inverse_transform, log_likelihood, save/load) |
| `src/models/ppca_classifier.py` | PPCALogisticClassifier and PPCAMixtureClassifier |
| `src/models/train_ppca.py` | Training script with component sweep |
| `src/preprocessing/augment.py` | Landmark-level augmentation functions |
| `scripts/visualize_ppca.py` | Scree plot, 2D projection, confusion matrix |
| `tests/test_ppca.py` | Unit tests for PPCA on synthetic data |

### Modified files

| File | Change |
|------|--------|
| `src/preprocessing/dataset.py` | Replace 80/20 split with stratified 60/20/20; add `split="test"`; add `augment` flag; add `get_arrays()` helper for sklearn-style consumption |
| `src/models/evaluate.py` | Add `--model ppca` branch; report top-1, per-class F1, confusion matrix |
| `Makefile` | Add `train-ppca`, `evaluate-ppca`, `visualize-ppca` targets |
| `requirements.txt` | Add `scikit-learn`, `matplotlib`, `seaborn` (already present?) |
| `README.md` | New "Phase 1b: PPCA Pipeline" section |

---

## PPCA Implementation

Closed-form via SVD (faster, deterministic) for v1. EM as a fallback /
alternative if we need to handle missing data later.

**Algorithm (closed-form, Tipping & Bishop §3.2):**

Given centered data **X** ∈ ℝ^(N×D) with sample covariance **S** = (1/N)X^T X:

1. Eigendecompose **S** = **UΛU**^T.
2. Take top *q* eigenvalues λ_1 ≥ ... ≥ λ_q and corresponding **U**_q.
3. Estimate noise variance: σ² = (1/(D−q)) Σ_{j=q+1}^{D} λ_j.
4. Loading matrix: **W** = **U**_q (**Λ**_q − σ²**I**)^(1/2).
5. Latent posterior mean (transform): **z** = **M**^(−1) **W**^T (**x** − **μ**), where **M** = **W**^T**W** + σ²**I**.

**Log-likelihood under PPCA** (used for Mixture-of-PPCA classifier):

log p(x) = −(1/2)[D·log(2π) + log|**C**| + (x−μ)^T **C**^(−1) (x−μ)]

where **C** = **WW**^T + σ²**I**. Use Woodbury identity for efficient inversion
when D > q.

---

## Training Procedure

1. Load train/val/test arrays from `ASLAlphabetDataset` (60/20/20, stratified).
2. Standardize features: zero mean, unit variance per dimension (fit on train).
3. **Sweep n_components ∈ {8, 12, 16, 20, 30}** on training set:
   - Fit PPCA(n) on train.
   - Transform train and val.
   - Fit `LogisticRegression(max_iter=1000)` on train PPCA features.
   - Record val accuracy.
4. Pick best n_components.
5. Refit PPCA + classifier with augmentation enabled on train.
6. **Test set evaluated once.**
7. Save artifacts to `experiments/ppca_<timestamp>/`:
   - `ppca_model.npz` (W, μ, σ², n_components, scaler params)
   - `classifier.pkl` (sklearn LogisticRegression)
   - `metrics.json` (val sweep, final test metrics)
   - `class_map.json`

Expected results: ≥85% val/test accuracy on landmarks (similar to MLP), with
the win showing up on **video inference** thanks to augmentation + denoising.

---

## Evaluation

**Phase 1b primary metrics:**
- Top-1 accuracy on test split (target ≥85%)
- Per-class F1 (especially low-frequency confusion: M/N, U/V, etc.)
- Confusion matrix
- Component sweep curve (val accuracy vs. n_components)

**Comparisons:**
1. MLP baseline (existing) vs. PPCA + LogReg
2. PPCA + LogReg vs. Mixture-of-PPCAs (stretch)
3. Static image test accuracy vs. video pipeline accuracy (the real goal)

**Visualizations for the report:**
- Scree plot — explained variance vs. n_components
- 2D scatter of letters in first two PPCA components, colored by class
- Reconstructed landmarks (overlay original vs. PPCA-reconstructed hand skeleton)
- Confusion matrix heatmap

---

## Implementation Order

1. **PPCA class** + unit tests on synthetic Gaussian data
2. **Dataset changes** (60/20/20 split, augmentation hook, get_arrays helper)
3. **Augmentation module** (jitter, rotate, mirror)
4. **PPCA classifier** (LogReg head + Mixture stretch)
5. **Training script** (sweep + final fit)
6. **Evaluation script update**
7. **Visualizations**
8. **Makefile + README + requirements**

Estimated effort: ~2-3 sessions of focused work.

---

## Out of Scope (for now)

- Pixel-space PPCA (eigenhands)
- Phase 2 LSTM / WLASL — separate effort
- Phase 3 LLM agent — separate effort
- Replacing video segmenter / keyframe extractor (orthogonal, upstream)
- Image-level augmentation requiring landmark re-extraction

---

## References

1. Tipping, M. E., & Bishop, C. M. (1999). *Probabilistic Principal Component Analysis.* JRSS-B 61(3): 611–622.
2. Tipping, M. E., & Bishop, C. M. (1999). *Mixtures of Probabilistic Principal Component Analyzers.* Neural Computation 11(2): 443–482.
3. Abeyta et al. *ASL Alphabet Dataset.* Kaggle, 2018.
4. Google. *MediaPipe Hands.*
