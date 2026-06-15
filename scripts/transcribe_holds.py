"""Per-frame classifier + hold-based decoder.

Walks through a video frame-by-frame, runs a per-frame letter classifier
(RF, MPPCA, SVM, or PPCA), groups consecutive frames with consistent
predictions into "holds", and outputs one letter per stable hold.

Why this works for slow / continuous fingerspelling like an alphabet recital:
each letter is held for many frames; majority vote within a hold is robust
to per-frame jitter; the only failure modes are very short letters (J, Z
trace motion) or letters that look like neighbors (M↔N).

Usage:
    make transcribe-holds VIDEO=example_videos/alphabet.mp4 EMISSION=experiments/rf_balanced
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


def extract_landmark_stream(video_path):
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
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
            out.append(last.copy())
        elif last is not None:
            out.append(last.copy())
    cap.release()
    landmarker.close()
    return (np.stack(out) if out else None), fps


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


def collapse_adjacent_dups(holds):
    """If two consecutive holds have the same letter (split by a brief noise
    run), merge them in the output sequence."""
    out = []
    for h in holds:
        if not out or out[-1] != h[0]:
            out.append(h[0])
    return "".join(out)


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
    parser.add_argument("--smooth-window", type=int, default=11,
                        help="Per-frame majority-vote smoothing (odd integer)")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: video not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting landmarks from {args.input}")
    X, fps = extract_landmark_stream(args.input)
    if X is None:
        print("ERROR: no frames produced landmarks", file=sys.stderr)
        sys.exit(1)
    print(f"  T={len(X)} frames @ {fps:.1f} fps")

    if args.mppca:
        pred_letters, proba, idx_to_letter = predict_per_frame_mppca(args.emission_dir, X)
    else:
        pred_letters, proba, idx_to_letter = predict_per_frame(args.emission_dir, X)

    min_hold = max(1, int(args.min_hold_seconds * fps))
    print(f"  min_hold_frames = {min_hold} ({args.min_hold_seconds:.2f}s)")
    print(f"  smoothing_window = {args.smooth_window}")

    holds = find_holds(pred_letters, min_hold, smooth_window=args.smooth_window)
    print(f"\nDetected {len(holds)} holds:")
    for letter, s, e in holds:
        print(f"  frames {s:5d}-{e:5d}  ({e-s:4d} frames, {(e-s)/fps:.2f}s): {letter}")

    sequence = collapse_adjacent_dups(holds)
    print(f"\nDecoded sequence: {sequence}")
    print(f"  length: {len(sequence)}  unique: {len(set(sequence))}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "video": args.input,
                "decoded": sequence,
                "n_frames": int(len(X)),
                "fps": float(fps),
                "min_hold_frames": int(min_hold),
                "smooth_window": int(args.smooth_window),
                "holds": [
                    {"letter": l, "start": int(s), "end": int(e), "duration": int(e - s)}
                    for l, s, e in holds
                ],
            }, f, indent=2)
        print(f"  Wrote {args.out_json}")


if __name__ == "__main__":
    main()
