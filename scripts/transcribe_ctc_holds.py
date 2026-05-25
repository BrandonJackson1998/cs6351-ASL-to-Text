"""CTC-as-per-frame letter classifier with hold-based decoding.

Same hold decoder as transcribe_holds.py, but the per-frame letter source
is the CTC v2 LSTM. We ignore the blank class and argmax over the 26 letter
classes only — turning CTC into a per-frame classifier for direct comparison
with the Random Forest pipeline.

This is a different question than the standard CTC sequence decoder asks:
  - Standard CTC sequence decode: "what is the most likely letter sequence?"
  - This script: "what is the most likely letter at each frame, ignoring
    the model's blank vote?"
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.preprocessing.combined_dataset import NUM_CLASSES
from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_landmark_stream(video_path, fill_with_last=True):
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
        elif fill_with_last and last is not None:
            out.append(last.copy())
    cap.release()
    landmarker.close()
    return (np.stack(out) if out else None), fps


def predict_per_frame_ctc_letters_only(checkpoint, X, device):
    """Run CTC v2 on a sequence and return per-frame top letter (ignoring blank).

    Returns (pred_letters: (T,) array of 'A'..'Z',
             pred_probs:   (T,) array of softmax probabilities for the chosen letter)
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = CTCLSTMv2(
        input_dim=63,
        hidden_dim=a.get("hidden_dim", 256),
        num_layers=a.get("num_layers", 3),
        dropout=a.get("dropout", 0.3),
        num_classes=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = torch.from_numpy(X.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([len(X)], dtype=torch.long)
    with torch.no_grad():
        log_probs = model(x, lengths).squeeze(1).cpu().numpy()
    probs = np.exp(log_probs)

    # Slot 0 is CTC blank. Argmax over slots 1..26 only.
    letter_probs = probs[:, 1:]  # (T, 26)
    pred_idx = letter_probs.argmax(axis=1)  # (T,) in 0..25
    pred_letters = np.array([_LETTERS[i] for i in pred_idx])
    pred_top = letter_probs[np.arange(len(pred_idx)), pred_idx]
    return pred_letters, pred_top


def smooth_majority(letters, window=11):
    T = len(letters)
    half = window // 2
    out = []
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out.append(Counter(letters[lo:hi]).most_common(1)[0][0])
    return np.array(out)


def find_holds(pred_letters, min_hold_frames):
    runs = []
    prev = None
    start = 0
    for t, p in enumerate(pred_letters):
        if p != prev:
            if prev is not None:
                runs.append((prev, start, t))
            prev = p
            start = t
    runs.append((prev, start, len(pred_letters)))
    return [(l, s, e) for l, s, e in runs if (e - s) >= min_hold_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--min-hold-seconds", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=11)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CTC v2 from {args.checkpoint}")
    pred_letters, pred_probs = predict_per_frame_ctc_letters_only(args.checkpoint, X, device)

    smoothed = smooth_majority(pred_letters, window=args.smooth_window)
    min_hold = max(1, int(args.min_hold_seconds * fps))
    print(f"  min_hold_frames = {min_hold} ({args.min_hold_seconds:.2f}s)")
    print(f"  smoothing_window = {args.smooth_window}")

    holds = find_holds(smoothed, min_hold)
    print(f"\nDetected {len(holds)} holds:")
    for letter, s, e in holds:
        print(f"  frames {s:5d}-{e:5d}  ({e-s:4d} frames, {(e-s)/fps:.2f}s): {letter}")

    # Collapse adjacent duplicates (rare)
    seq = []
    for l, _, _ in holds:
        if not seq or seq[-1] != l:
            seq.append(l)
    decoded = "".join(seq)

    print(f"\nDecoded sequence: {decoded}")
    print(f"  length: {len(decoded)}  unique: {len(set(decoded))}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "video": args.input,
                "decoded": decoded,
                "n_frames": int(len(X)),
                "fps": float(fps),
                "holds": [
                    {"letter": l, "start": int(s), "end": int(e), "duration": int(e - s)}
                    for l, s, e in holds
                ],
            }, f, indent=2)
        print(f"  Wrote {args.out_json}")


if __name__ == "__main__":
    main()
