VENV = .virtual_environment
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

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
	$(PYTHON) scripts/download_data.py --dataset alphabet

extract-landmarks:
	$(PYTHON) src/preprocessing/landmark_extractor.py --dataset alphabet

train-baseline:
	$(PYTHON) src/models/train_mlp.py

evaluate-baseline:
	$(PYTHON) src/models/evaluate.py --model mlp --checkpoint experiments/latest/best_model.pt

# ============================================================
# Phase 2: Dynamic Word Recognition
# ============================================================

download-wlasl:
	$(PYTHON) scripts/download_data.py --dataset wlasl --num-words 100

extract-video-landmarks:
	$(PYTHON) src/preprocessing/landmark_extractor.py --dataset wlasl

train-lstm:
	$(PYTHON) src/models/train_lstm.py

evaluate-lstm:
	$(PYTHON) src/models/evaluate.py --model lstm --checkpoint experiments/latest/best_model.pt

# ============================================================
# Phase 3: Sentence Assembly
# ============================================================

translate:
	$(PYTHON) src/agents/gloss_translator.py --glosses "$(GLOSSES)"

# ============================================================
# End-to-End Pipeline
# ============================================================

demo:
	$(PYTHON) scripts/demo.py

demo-webcam:
	$(PYTHON) scripts/demo_webcam.py

# ============================================================
# Video Segmentation
# ============================================================

segment-video:
	$(PYTHON) src/preprocessing/video_segmenter.py --input "$(VIDEO)" --output-dir segments/

segment-video-viz:
	$(PYTHON) src/preprocessing/video_segmenter.py --input "$(VIDEO)" --output-dir segments/ --visualize

extract-keyframes:
	PYTHONPATH=. $(PYTHON) src/preprocessing/keyframe_extractor.py --input "$(VIDEO)" --output-dir segments/

visualize-segmentation:
	$(PYTHON) scripts/visualize_segmentation.py --input "$(VIDEO)" --output-dir segments/

analyze-video:
	@echo "=== Segmenting video ==="
	$(PYTHON) src/preprocessing/video_segmenter.py --input "$(VIDEO)" --output-dir segments/
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

.PHONY: venv install install-mac install-pip download-alphabet extract-landmarks \
	train-baseline evaluate-baseline download-wlasl extract-video-landmarks \
	train-lstm evaluate-lstm translate demo demo-webcam \
	segment-video segment-video-viz extract-keyframes \
	visualize-segmentation analyze-video test clean
