"""Per-frame classifier + hold decoder + word segmentation.

Extends transcribe_holds.py with word boundary detection via:
- Hand-drop gaps (when MediaPipe loses hand or confidence drops)
- Pause duration (long gaps between letter holds)

Output format: letter sequences with | marking word boundaries
Example: FIG FIG | DATE DATE | LIME LIME | GUAVA

Usage:
    python scripts/transcribe_with_segmentation.py \
        --input test_ensemble.mp4 \
        --emission-dir experiments/rf_balanced \
        --pause-threshold 1.0
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import mediapipe as mp

from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_landmark_stream_with_confidence(video_path):
    """Extract landmarks and track hand presence/confidence for segmentation."""
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarks = []
    hand_present = []
    last = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        r = landmarker.detect(mp_image)

        if r.hand_landmarks:
            lm = r.hand_landmarks[0]
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
            last = normalize_landmarks(raw)
            landmarks.append(last.copy())
            hand_present.append(True)
        else:
            # No hand detected — use last valid landmarks but mark as absent
            if last is not None:
                landmarks.append(last.copy())
            else:
                # Very first frames with no hand — skip
                continue
            hand_present.append(False)

    cap.release()
    landmarker.close()
    return (np.stack(landmarks) if landmarks else None), np.array(hand_present), fps


def predict_per_frame(emission_dir, X):
    """Return (preds, proba) where preds is (T,) class indices and proba is (T, n_classes)."""
    with open(os.path.join(emission_dir, "model.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(emission_dir, "class_map.json")) as f:
        class_map = json.load(f)
    scaler_path = os.path.join(emission_dir, "scaler.npz")
    if os.path.isfile(scaler_path):
        sd = np.load(scaler_path)
        X = (X - sd["mean"]) / sd["scale"]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        preds = clf.classes_[np.argmax(proba, axis=1)]
    else:
        proba = None
        preds = clf.predict(X)

    idx_to_letter = {int(v): k.upper() for k, v in class_map.items()}
    pred_letters = np.array([idx_to_letter.get(int(p), "?") for p in preds])
    return pred_letters, proba, idx_to_letter


def predict_per_frame_mppca(emission_dir, X):
    """Same as above but for MPPCAClassifier (different load API)."""
    from src.models.mppca import MPPCAClassifier
    clf = MPPCAClassifier.load(emission_dir)
    proba = clf.predict_proba(X)
    preds = clf.classes_[np.argmax(proba, axis=1)]

    cm_path = os.path.join(emission_dir, "class_map.json")
    if not os.path.isfile(cm_path):
        cm_path = os.path.join(os.path.dirname(emission_dir), "class_map.json")
    with open(cm_path) as f:
        class_map = json.load(f)
    idx_to_letter = {int(v): k.upper() for k, v in class_map.items()}
    pred_letters = np.array([idx_to_letter.get(int(p), "?") for p in preds])
    return pred_letters, proba, idx_to_letter


def find_holds(pred_letters, min_hold_frames, smooth_window=5):
    """Group consecutive frames with the same letter into holds.

    Smoothing: replace each frame's letter with the majority letter in a
    sliding window of size `smooth_window`. Reduces single-frame flicker
    inside a hold.
    """
    T = len(pred_letters)
    if T == 0:
        return []

    if smooth_window > 1:
        half = smooth_window // 2
        smoothed = []
        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            window = pred_letters[lo:hi]
            most_common = Counter(window).most_common(1)[0][0]
            smoothed.append(most_common)
        pred_letters = np.array(smoothed)

    runs = []
    prev = None
    start = 0
    for t in range(T):
        if pred_letters[t] != prev:
            if prev is not None:
                runs.append((prev, start, t))
            prev = pred_letters[t]
            start = t
    runs.append((prev, start, T))

    holds = [(letter, s, e) for letter, s, e in runs if (e - s) >= min_hold_frames]
    return holds


def segment_by_gaps(holds, hand_present, fps, pause_threshold_seconds):
    """Segment holds into words based on hand-drop gaps and long pauses.

    Returns: list of word segments, each is a list of (letter, start, end) tuples
    """
    if not holds:
        return []

    pause_threshold_frames = int(pause_threshold_seconds * fps)

    segments = []
    current_segment = []

    for i, (letter, start, end) in enumerate(holds):
        # Check if hand was dropped between previous hold and this one
        if i > 0:
            prev_end = holds[i-1][2]
            gap_start = prev_end
            gap_end = start

            # Hand-drop gap: check if hand was absent during gap
            gap_hand_present = hand_present[gap_start:gap_end]
            hand_dropped = len(gap_hand_present) > 0 and np.mean(gap_hand_present) < 0.5

            # Long pause gap
            gap_length = gap_end - gap_start
            long_pause = gap_length >= pause_threshold_frames

            # Start new segment if hand dropped OR long pause
            if hand_dropped or long_pause:
                if current_segment:
                    segments.append(current_segment)
                current_segment = []

        current_segment.append((letter, start, end))

    # Add final segment
    if current_segment:
        segments.append(current_segment)

    return segments


def format_segmented_output(segments):
    """Format segments as string with | separators."""
    words = []
    for segment in segments:
        # Collapse consecutive duplicates within each segment
        letters = []
        for letter, _, _ in segment:
            if not letters or letters[-1] != letter:
                letters.append(letter)
        words.append("".join(letters))
    return " | ".join(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--emission-dir", required=True,
                        help="Directory with model.pkl + class_map.json (RF/SVM/PPCA) "
                             "or MPPCAClassifier saved dir")
    parser.add_argument("--mppca", action="store_true",
                        help="Set if emission-dir is an MPPCAClassifier (uses .load() API)")
    parser.add_argument("--min-hold-seconds", type=float, default=0.5,
                        help="Holds shorter than this are treated as noise")
    parser.add_argument("--pause-threshold", type=float, default=1.0,
                        help="Gaps longer than this (seconds) mark word boundaries")
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="Per-frame majority-vote smoothing (odd integer)")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting landmarks from {args.input}")
    X, hand_present, fps = extract_landmark_stream_with_confidence(args.input)
    if X is None:
        print("ERROR: no frames produced landmarks", file=sys.stderr)
        sys.exit(1)
    print(f"  T={len(X)} frames @ {fps:.1f} fps")
    print(f"  Hand present: {np.sum(hand_present)}/{len(hand_present)} frames ({100*np.mean(hand_present):.1f}%)")

    if args.mppca:
        pred_letters, proba, idx_to_letter = predict_per_frame_mppca(args.emission_dir, X)
    else:
        pred_letters, proba, idx_to_letter = predict_per_frame(args.emission_dir, X)

    min_hold = max(1, int(args.min_hold_seconds * fps))
    print(f"  min_hold_frames = {min_hold} ({args.min_hold_seconds:.2f}s)")
    print(f"  pause_threshold = {int(args.pause_threshold * fps)} frames ({args.pause_threshold:.2f}s)")
    print(f"  smoothing_window = {args.smooth_window}")

    holds = find_holds(pred_letters, min_hold, smooth_window=args.smooth_window)
    print(f"\nDetected {len(holds)} holds")

    segments = segment_by_gaps(holds, hand_present, fps, args.pause_threshold)
    print(f"Detected {len(segments)} word segments:")
    for i, segment in enumerate(segments, 1):
        letters = "".join([l for l, _, _ in segment])
        duration = sum(e - s for _, s, e in segment) / fps
        print(f"  Segment {i}: {letters} ({duration:.2f}s, {len(segment)} holds)")

    output = format_segmented_output(segments)
    print(f"\nSegmented output:")
    print(f"  {output}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "video": args.input,
                "segmented": output,
                "n_frames": int(len(X)),
                "fps": float(fps),
                "n_segments": len(segments),
                "segments": [
                    {
                        "letters": "".join([l for l, _, _ in seg]),
                        "holds": [
                            {"letter": l, "start": int(s), "end": int(e), "duration": int(e - s)}
                            for l, s, e in seg
                        ]
                    }
                    for seg in segments
                ],
            }, f, indent=2)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
