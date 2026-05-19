VENV = .virtual_environment
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Set to 1 to show MediaPipe's C-level logs (suppressed by default).
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
# Phase 1: Static Fingerspelling
# ============================================================

download-alphabet:
	$(PYTHON) scripts/download_data.py --dataset alphabet --output-dir data

download-wlasl-words:
	$(PYTHON) scripts/download_data.py --dataset wlasl --output-dir data --num-words $(or $(NUM_WORDS),100)

download-all:
	$(PYTHON) scripts/download_data.py --dataset all --output-dir data --num-words $(or $(NUM_WORDS),100)

extract-landmarks:
	$(PYTHON) -m src.preprocessing.landmark_extractor --dataset alphabet

train-baseline:
	$(PYTHON) -m src.models.train_mlp

evaluate-baseline:
	$(PYTHON) -m src.models.evaluate --model mlp --checkpoint experiments/latest/best_model.pt

# ============================================================
# Phase 2: Dynamic Word Recognition
# ============================================================

download-wlasl:
	$(PYTHON) scripts/download_data.py --dataset wlasl --output-dir data --num-words 100

extract-video-landmarks:
	$(PYTHON) -m src.preprocessing.landmark_extractor --dataset wlasl

train-lstm:
	$(PYTHON) -m src.models.train_lstm

evaluate-lstm:
	$(PYTHON) -m src.models.evaluate --model lstm --checkpoint experiments/latest/best_model.pt

# ============================================================
# Phase 3: Sentence Assembly
# ============================================================

translate:
	$(PYTHON) -m src.agents.gloss_translator --glosses "$(GLOSSES)"

# ============================================================
# End-to-End Pipeline
# ============================================================

demo:
	$(PYTHON) scripts/demo.py

demo-webcam:
	$(PYTHON) scripts/demo_webcam.py

# ============================================================
# Frame Analysis & Sequence
# ============================================================

analyze-frame:
	@PYTHONPATH=. GLOG_minloglevel=3 TF_CPP_MIN_LOG_LEVEL=3 MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/analyze_frame.py \
		--frame "$(FRAME)" \
		--checkpoint "$(or $(CHECKPOINT),experiments/latest/best_model.pt)" \
		$(if $(VERBOSE),--verbose,)

string-deltas:
	@PYTHONPATH=. MEDIAPIPE_VERBOSE=$(MEDIAPIPE_VERBOSE) $(PYTHON) scripts/string_deltas.py \
		--input "$(VIDEO)" \
		--checkpoint "$(or $(CHECKPOINT),experiments/latest/best_model.pt)" \
		--output-dir segments/ \
		--confidence-threshold $(or $(CONFIDENCE),0.0) \
		$(if $(VERBOSE),--verbose,)

# ============================================================
# Video Segmentation
# ============================================================

segment-video:
	$(PYTHON) -m src.preprocessing.video_segmenter --input "$(VIDEO)" --output-dir segments/

segment-video-viz:
	$(PYTHON) -m src.preprocessing.video_segmenter --input "$(VIDEO)" --output-dir segments/ --visualize

extract-keyframes:
	PYTHONPATH=. $(PYTHON) src/preprocessing/keyframe_extractor.py --input "$(VIDEO)" --output-dir segments/

visualize-segmentation:
	$(PYTHON) scripts/visualize_segmentation.py --input "$(VIDEO)" --output-dir segments/

analyze-video:
	@echo "=== Segmenting video ==="
	$(PYTHON) -m src.preprocessing.video_segmenter --input "$(VIDEO)" --output-dir segments/
	@echo "\n=== Extracting keyframes ==="
	PYTHONPATH=. $(PYTHON) src/preprocessing/keyframe_extractor.py --input "$(VIDEO)" --output-dir segments/
	@echo "\n=== Generating visualization ==="
	$(PYTHON) scripts/visualize_segmentation.py --input "$(VIDEO)" --output-dir segments/
	@echo "\n=== Done ==="

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
	download-alphabet download-wlasl download-wlasl-words download-all \
	extract-landmarks train-baseline evaluate-baseline \
	download-wlasl extract-video-landmarks \
	train-lstm evaluate-lstm translate demo demo-webcam \
	analyze-frame string-deltas \
	segment-video segment-video-viz extract-keyframes \
	visualize-segmentation analyze-video test clean
