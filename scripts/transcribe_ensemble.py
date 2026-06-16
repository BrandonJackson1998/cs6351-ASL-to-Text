#!/usr/bin/env python3
"""
Ensemble transcription combining Random Forest + PPCA predictions.

Simple weighted voting: average probability distributions from both models
and use the highest confidence prediction.

Expected improvement: 83% (RF alone) → 85-87% (ensemble)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks
from src.models.ppca_classifier import PPCAMixtureClassifier


def load_models(rf_path: Path, ppca_dir: Path):
    """Load both RF and PPCA models."""
    print(f"Loading RF model from {rf_path}")
    with open(rf_path, "rb") as f:
        rf_model = pickle.load(f)

    print(f"Loading PPCA model from {ppca_dir}/mixture/")
    ppca_model = PPCAMixtureClassifier.load(str(ppca_dir / "mixture"))

    return rf_model, ppca_model


def extract_landmarks(video_path: Path):
    """Extract landmarks from video frames."""
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    landmarks = []
    frame_indices = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = __import__('mediapipe').Image(
            image_format=__import__('mediapipe').ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_img)

        if result.hand_landmarks:
            # Use first detected hand
            lm = result.hand_landmarks[0]
            raw = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
            lm_array = normalize_landmarks(raw)
            landmarks.append(lm_array)
            frame_indices.append(frame_idx)

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(landmarks)} frames with landmarks from {frame_idx} total frames")

    return landmarks, frame_indices


def ensemble_predict(rf_model, ppca_model, features: np.ndarray, rf_weight=0.6):
    """
    Ensemble prediction using weighted average of probabilities.

    Args:
        rf_model: Random Forest classifier
        ppca_model: PPCA mixture classifier
        features: Feature vector (63D landmarks)
        rf_weight: Weight for RF predictions (PPCA gets 1-rf_weight)

    Returns:
        Predicted class index
        Combined probability distribution
    """
    # Get probability distributions from both models
    rf_proba = rf_model.predict_proba(features.reshape(1, -1))[0]
    ppca_proba = ppca_model.predict_proba(features.reshape(1, -1))[0]

    # Weighted average
    ensemble_proba = rf_weight * rf_proba + (1 - rf_weight) * ppca_proba

    # Return highest confidence prediction
    pred_class = ensemble_proba.argmax()

    return pred_class, ensemble_proba


def collapse_holds(predictions, class_map, min_hold_frames=3):
    """
    Collapse consecutive identical predictions into holds.

    Args:
        predictions: List of class indices
        class_map: Dict mapping class index to letter
        min_hold_frames: Minimum frames to count as a hold

    Returns:
        String of collapsed letters
    """
    if not predictions:
        return ""

    result = []
    current_letter = predictions[0]
    hold_count = 1

    for pred in predictions[1:]:
        if pred == current_letter:
            hold_count += 1
        else:
            # End of hold
            if hold_count >= min_hold_frames:
                result.append(class_map[current_letter])
            current_letter = pred
            hold_count = 1

    # Handle last hold
    if hold_count >= min_hold_frames:
        result.append(class_map[current_letter])

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble RF + PPCA transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default models
  python scripts/transcribe_ensemble.py --input video.mp4

  # Custom RF weight (higher = trust RF more)
  python scripts/transcribe_ensemble.py --input video.mp4 --rf-weight 0.7

  # Specify models explicitly
  python scripts/transcribe_ensemble.py --input video.mp4 \\
      --rf-model experiments/rf_balanced/model.pkl \\
      --ppca-model experiments/ppca_20260615_212735/mixture_model.pkl
"""
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input video file"
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
        default=Path("experiments/rf_balanced/model.pkl"),
        help="Random Forest model path (default: experiments/rf_balanced/model.pkl)"
    )

    parser.add_argument(
        "--ppca-model",
        type=Path,
        default=Path("experiments/ppca_20260615_212735"),
        help="PPCA model directory (default: experiments/ppca_20260615_212735)"
    )

    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.6,
        help="Weight for RF predictions (0-1, default: 0.6). PPCA gets (1 - rf_weight)"
    )

    parser.add_argument(
        "--min-hold",
        type=int,
        default=3,
        help="Minimum consecutive frames to count as a letter hold (default: 3)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output text file (default: print to stdout)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-frame predictions"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"Error: Video file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.rf_model.exists():
        print(f"Error: RF model not found: {args.rf_model}", file=sys.stderr)
        return 1

    if not (args.ppca_model / "mixture").exists():
        print(f"Error: PPCA model directory not found: {args.ppca_model}/mixture", file=sys.stderr)
        return 1

    if not 0 <= args.rf_weight <= 1:
        print(f"Error: --rf-weight must be between 0 and 1, got {args.rf_weight}", file=sys.stderr)
        return 1

    # Load models
    rf_model, ppca_model = load_models(args.rf_model, args.ppca_model)

    # Load class map from RF model directory
    class_map_path = args.rf_model.parent / "class_map.json"
    import json
    with open(class_map_path) as f:
        letter_to_idx = json.load(f)
        # Reverse: idx to letter
        class_map = {v: k for k, v in letter_to_idx.items()}

    print(f"Loaded {len(class_map)} classes: {sorted(class_map.values())}")
    print(f"Ensemble weights: RF={args.rf_weight:.1f}, PPCA={1-args.rf_weight:.1f}")
    print()

    # Extract landmarks
    landmarks, frame_indices = extract_landmarks(args.input)

    if not landmarks:
        print("Error: No hand landmarks detected in video", file=sys.stderr)
        return 1

    # Run ensemble predictions
    print(f"Running ensemble prediction on {len(landmarks)} frames...")
    predictions = []

    for i, lm in enumerate(landmarks):
        pred_class, proba = ensemble_predict(rf_model, ppca_model, lm, args.rf_weight)
        predictions.append(pred_class)

        if args.verbose:
            letter = class_map[pred_class]
            confidence = proba[pred_class]
            print(f"Frame {frame_indices[i]:4d}: {letter} ({confidence:.3f})")

    # Collapse holds
    transcription = collapse_holds(predictions, class_map, args.min_hold)

    print()
    print("=" * 60)
    print(f"TRANSCRIPTION: {transcription}")
    print("=" * 60)
    print(f"Total frames: {len(predictions)}")
    print(f"Letters detected: {len(transcription)}")

    # Save to file if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(transcription)
        print(f"\nSaved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
