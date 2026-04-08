# ASL Video-to-Text Translation

**CS-6351: AI & ML -- Course Project**

An end-to-end machine learning system that translates American Sign Language (ASL) video into natural English text. The pipeline combines computer vision (hand/pose detection), sequence modeling (temporal sign recognition), and natural language processing (grammar translation via LLM agent).

**Team:** Brandon Jackson & Kevin Bateman

---

## Architecture

```
Video -> Frame Extraction -> Hand/Pose Detection -> Sign Classification -> LLM Translation -> English Text
```

## Project Phases

| Phase | Goal | Model | Dataset |
|-------|------|-------|---------|
| **1: Foundation** | Static fingerspelling (image -> letter) | MLP on hand landmarks | ASL Alphabet (87k images, 29 classes) |
| **2: Video** | Dynamic word recognition (video -> word) | LSTM on temporal landmarks | WLASL subset (100-300 words) |
| **3: Sentences** | Gloss sequence -> English sentence | Claude API agent | -- |
| **4: Bonus** | Real-time webcam, ASL-to-Voice (TTS) | -- | -- |

## Quick Start

### Setup

```bash
# macOS
make install-mac

# Linux
make install
```

### Phase 1: Fingerspelling Classifier

```bash
make download-alphabet       # Download ASL Alphabet dataset from Kaggle
make extract-landmarks       # Extract MediaPipe hand landmarks
make train-baseline          # Train MLP classifier
make evaluate-baseline       # Evaluate on test set
```

### Phase 2: Word Recognition

```bash
make download-wlasl          # Download WLASL video subset
make extract-video-landmarks # Extract per-frame holistic landmarks
make train-lstm              # Train LSTM sequence model
make evaluate-lstm           # Evaluate on test set
```

### Phase 3: Sentence Assembly

```bash
make translate GLOSSES="STORE,I,GO,YESTERDAY"   # Translate glosses to English
make demo                                        # Launch Gradio web demo
```

## Project Structure

```
cs6351-ASL-to-Text/
├── src/
│   ├── preprocessing/
│   │   ├── landmark_extractor.py   # MediaPipe hand/holistic landmark extraction
│   │   └── dataset.py              # PyTorch dataset classes
│   ├── models/
│   │   ├── mlp.py                  # Phase 1: MLP fingerspelling classifier
│   │   ├── lstm.py                 # Phase 2: LSTM word sign recognizer
│   │   ├── train_mlp.py            # MLP training script
│   │   ├── train_lstm.py           # LSTM training script
│   │   └── evaluate.py             # Model evaluation & metrics
│   ├── agents/
│   │   ├── gloss_translator.py     # Phase 3: Claude API gloss-to-English translation
│   │   └── pipeline.py             # End-to-end video-to-text pipeline
│   └── utils/
├── scripts/
│   ├── download_data.py            # Dataset download helpers
│   ├── demo.py                     # Gradio web demo
│   └── demo_webcam.py              # Real-time webcam demo
├── data/                           # Datasets (not tracked in git)
│   ├── asl_alphabet/               # ASL Alphabet images (Phase 1)
│   └── wlasl/                      # WLASL videos & landmarks (Phase 2)
├── experiments/                    # Training outputs & checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
├── docs/                           # Documentation
├── Makefile                        # Build & run commands
├── requirements.txt                # Python dependencies
└── README.md
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Pose/Hand Extraction | MediaPipe Holistic |
| Model Training | PyTorch |
| Video Processing | OpenCV |
| Agent / Translation | Claude API (Anthropic) |
| Demo UI | Gradio |
| Training Infra | Google Colab / University Cluster |

## Datasets

- **ASL Alphabet** -- 87k static images, 29 classes (A-Z + SPACE, DELETE, NOTHING). [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **WLASL** -- Word-Level ASL, ~21k videos, 2000 words. [Project Page](https://dxli94.github.io/WLASL/)

## Evaluation Metrics

- **Phase 1:** Top-1 accuracy (target >= 90%)
- **Phase 2:** Top-1 and Top-5 word accuracy; per-class F1
- **Phase 3:** BLEU score; qualitative fluency review

## Team Responsibilities

| Area | Owner |
|------|-------|
| Data pipeline (video -> landmarks -> features), preprocessing, dataset management | Brandon Jackson |
| Model architecture, training loop, evaluation, LLM agent integration | Kevin Bateman |
| End-to-end integration, demo, evaluation analysis, final report | Both |

## References

1. Abeyta et al. *ASL Alphabet Dataset*. Kaggle, 2018.
2. Li et al. *Word-Level Deep Sign Language Recognition from Video*. WACV 2020.
3. Google. *MediaPipe Holistic*.
4. Adhikary. *Realtime Sign Language Detection Using LSTM Model*. GitHub.
5. Mistry. *ASL Recognition Using Deep Neural Networks*. GitHub.
