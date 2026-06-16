VENV = .virtual_environment
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Set MEDIAPIPE_VERBOSE=1 to see MediaPipe's C-level logs.
MEDIAPIPE_VERBOSE ?= 0

# ============================================================
# Environment Setup
# ============================================================

venv:
	python3 -m venv $(VENV)

install-pip: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-mac: venv
	brew install python@3.12 ffmpeg
	$(MAKE) install-pip

install: venv
	sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv ffmpeg
	$(MAKE) install-pip

# ============================================================
# Data Pipeline
# ============================================================

# Single command to (re)generate every dataset artifact the project depends on.
# Steps run in dependency order; each step skips if its output already exists.
# Total ~2 hours from scratch on CPU; ~13 GB disk.
build-data: download-alphabet extract-landmarks download-fswild extract-fswild train-ctc-v2 extract-fswild-letter-frames build-balanced-letter-frames
	@echo "\n=== Data pipeline complete ==="
	@echo "Artifacts:"
	@echo "  data/asl_alphabet/landmarks/        (Kaggle, per-letter)"
	@echo "  data/chicagofswild/landmarks/        (FSWild, per-sequence)"
	@echo "  data/chicagofswild/letter_frames/    (FSWild, forced-aligned per-letter)"
	@echo "  data/balanced_letter_frames/         (Balanced merged, 4k/letter)"
	@echo "  experiments/latest_ctc_v2/           (CTC v2 checkpoint)"

# Download Kaggle ASL Alphabet (87k static images, 26 letters, ~1 GB)
download-alphabet:
	$(PYTHON) scripts/download_data.py --dataset alphabet --output-dir data

# Extract MediaPipe Hand landmarks from Kaggle alphabet images
extract-landmarks:
	$(PYTHON) -m src.preprocessing.landmark_extractor --dataset alphabet

# Extract hand crop images (200x200) from Kaggle alphabet for CNN training
extract-hand-crops:
	$(PYTHON) scripts/extract_hand_crops.py \
		--dataset kaggle \
		--input data \
		--output data/hand_crops_kaggle \
		$(if $(SIZE),--size $(SIZE),)

# Download ChicagoFSWild from Kaggle and extract frames (~13 GB)
download-fswild:
	PYTHONPATH=. $(PYTHON) scripts/download_fswild.py

# Extract MediaPipe landmarks from FSWild video sequences (for CTC training)
extract-fswild:
	PYTHONPATH=. $(PYTHON) -m src.preprocessing.chicagofswild --frames-root data/chicagofswild --partition $(or $(PARTITION),all)

# Forced-align FSWild train via CTC, write per-letter frame buckets.
# Requires a trained CTC v2 checkpoint at experiments/latest_ctc_v2.
extract-fswild-letter-frames:
	PYTHONPATH=. $(PYTHON) -u -m src.preprocessing.fswild_letter_frames

# Merge + balance Kaggle and FSWild per-letter frames
build-balanced-letter-frames:
	PYTHONPATH=. $(PYTHON) -u -m src.preprocessing.balanced_letter_frames $(if $(TARGET),--target $(TARGET),)

# ============================================================
# Per-frame Letter Classifiers (trained on data/balanced_letter_frames/)
# ============================================================

# RANDOM FOREST -- the primary per-frame classifier (82.7% on FSWild test)
train-rf:
	PYTHONPATH=. $(PYTHON) -u -m src.models.train_rf $(if $(EXP),--exp-name $(EXP),) $(if $(DATA),--data-dir $(DATA),)

# MPPCA -- mixture of PPCAs per letter (78.3%)
train-mppca:
	PYTHONPATH=. $(PYTHON) -u -m src.models.train_mppca $(if $(EXP),--exp-name $(EXP),) $(if $(DATA),--data-dir $(DATA),) $(if $(KS),--ks $(KS),) $(if $(QS),--qs $(QS),)

# PPCA + Mixture-of-PPCAs -- Phase 1 baseline (75.8% on FSWild, 98.2% on Kaggle)
train-ppca:
	$(PYTHON) -m src.models.train_ppca $(if $(MIXTURE),--mixture,) $(if $(COMPONENTS),--components $(COMPONENTS),) $(if $(AUG_FACTOR),--augment-factor $(AUG_FACTOR),)

refit-ppca-fswild:
	PYTHONPATH=. $(PYTHON) -u -m src.models.refit_ppca_fswild $(if $(EXP),--exp-name $(EXP),) $(if $(DATA),--data-dir $(DATA),)

# RBF-SVM -- ablation comparison (58% on FSWild)
train-svm:
	PYTHONPATH=. $(PYTHON) -u -m src.models.train_svm $(if $(EXP),--exp-name $(EXP),) $(if $(DATA),--data-dir $(DATA),)

# MLP -- original Phase 1 baseline
train-baseline:
	$(PYTHON) -m src.models.train_mlp

evaluate-baseline:
	$(PYTHON) -m src.models.evaluate --model mlp --checkpoint experiments/latest/best_model.pt

# Phase 1 PPCA evaluation
evaluate-ppca:
	$(PYTHON) -m src.models.evaluate --model ppca --checkpoint experiments/latest --split $(or $(SPLIT),test) $(if $(CM),--confusion-matrix $(CM),)

visualize-ppca:
	PYTHONPATH=. $(PYTHON) scripts/visualize_ppca.py --checkpoint experiments/latest

# ============================================================
# Sequence Model: CTC Bi-LSTM v2
# ============================================================

train-ctc-v2:
	$(PYTHON) -u -m src.models.train_ctc_v2 --epochs $(or $(EPOCHS),60) --patience $(or $(PATIENCE),8) $(if $(EXP),--exp-name $(EXP),)

# Single-shot test eval of CTC v2 on FSWild test (47.1% LER)
evaluate-ctc-v2:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_ctc_v2.py --checkpoint "$(or $(CHECKPOINT),experiments/latest_ctc_v2/best_model.pt)"

# CTC + per-frame emission fusion sweep (PPCA / Mixture-of-PPCAs)
evaluate-fusion:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_fusion.py $(if $(MIXTURE),--mixture-dir $(MIXTURE),) $(if $(CLASS_MAP),--class-map $(CLASS_MAP),)

# CTC + Random Forest fusion sweep
evaluate-fusion-rf:
	PYTHONPATH=. $(PYTHON) scripts/evaluate_fusion_rf.py $(if $(RF_DIR),--rf-dir $(RF_DIR),)

# ============================================================
# Inference: video -> letters
# ============================================================

# PRIMARY: Random Forest hold decoder. Best for slow / OOD signing
# (e.g. alphabet recital, the Finger.mp4 fruit demo).
transcribe-holds:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/transcribe_holds.py \
		--input "$(VIDEO)" \
		--emission-dir "$(or $(EMISSION),experiments/rf_balanced)" \
		$(if $(MPPCA),--mppca,) \
		$(if $(MIN_HOLD),--min-hold-seconds $(MIN_HOLD),) \
		$(if $(SMOOTH),--smooth-window $(SMOOTH),) \
		$(if $(OUT_JSON),--out-json $(OUT_JSON),)

# ENSEMBLE: RF + PPCA weighted voting. Expected 2-4% accuracy improvement.
transcribe-ensemble:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/transcribe_ensemble.py \
		--input "$(VIDEO)" \
		$(if $(RF_MODEL),--rf-model $(RF_MODEL),) \
		$(if $(PPCA_MODEL),--ppca-model $(PPCA_MODEL),) \
		$(if $(RF_WEIGHT),--rf-weight $(RF_WEIGHT),) \
		$(if $(MIN_HOLD),--min-hold $(MIN_HOLD),) \
		$(if $(OUTPUT),--output $(OUTPUT),) \
		$(if $(VERBOSE),--verbose,)

# CTC v2 used as a per-frame letter classifier (argmax over letter classes,
# ignore blank), feeding into the same hold decoder as transcribe-holds.
# A direct apples-to-apples per-frame comparison vs Random Forest.
transcribe-ctc-holds:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/transcribe_ctc_holds.py \
		--input "$(VIDEO)" \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(MIN_HOLD),--min-hold-seconds $(MIN_HOLD),) \
		$(if $(SMOOTH),--smooth-window $(SMOOTH),) \
		$(if $(OUT_JSON),--out-json $(OUT_JSON),)

# CTC v2 with optional per-frame emission fusion. Best for fast in-distribution
# fingerspelling (FSWild-style real words).
transcribe-v2-fusion:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/transcribe_v2_fusion.py \
		--input "$(VIDEO)" \
		--emission-dir "$(or $(EMISSION),experiments/rf_balanced)" \
		$(if $(LAM),--lam $(LAM),)

# Single-frame inference
analyze-frame:
	@PYTHONPATH=. GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/analyze_frame.py \
		--frame "$(FRAME)" \
		--checkpoint "$(or $(CHECKPOINT),experiments/latest/best_model.pt)" \
		$(if $(VERBOSE),--verbose,)

# ============================================================
# Visualization & Analysis
# ============================================================

# Side-by-side per-frame predictions: RF / MPPCA / PPCA / CTC.
# Output: segments/<video>/model_comparison_grid.png
model-comparison-grid:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/model_comparison_grid.py \
		--input "$(VIDEO)"

# ============================================================
# Utilities
# ============================================================

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: venv install install-mac install-pip \
	build-data download-alphabet extract-landmarks \
	download-fswild extract-fswild \
	extract-fswild-letter-frames build-balanced-letter-frames \
	train-rf train-mppca train-ppca refit-ppca-fswild train-svm \
	train-baseline evaluate-baseline evaluate-ppca visualize-ppca \
	train-ctc-v2 evaluate-ctc-v2 evaluate-fusion evaluate-fusion-rf \
	transcribe-holds transcribe-ctc-holds transcribe-v2-fusion analyze-frame \
	model-comparison-grid \
	test clean
