"""Visual side-by-side comparison of per-frame models on a video.

Pulls representative frames (one per RF-detected hold) and shows each one
with predictions from RF, MPPCA, PPCA-Mixture, and CTC v2 (per-frame argmax).

Output: a grid PNG where each tile is one frame with a label like:
   "frame 1234
    RF: C (0.89)
    MPPCA: C (0.71)
    PPCA: C (0.62)
    CTC: D (0.45)"
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
import matplotlib.pyplot as plt
import numpy as np
import torch
import mediapipe as mp

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.preprocessing.combined_dataset import NUM_CLASSES
from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_frames_and_landmarks(video_path):
    """Return (frames_bgr, landmarks (T,63), fps). Both same length T."""
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, lms = [], []
    last = None
    last_frame = None
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
            last_frame = frame
        if last is not None:
            lms.append(last.copy())
            frames.append(last_frame.copy() if last_frame is not None else frame)
    cap.release()
    landmarker.close()
    return frames, np.stack(lms), fps


def predict_proba_with_model(emission_dir, X, kind="sklearn"):
    """Return (preds_letters, top_proba, idx_to_letter) for the model in emission_dir."""
    if kind == "mppca":
        from src.models.mppca import MPPCAClassifier
        clf = MPPCAClassifier.load(emission_dir)
        proba = clf.predict_proba(X)
        preds = clf.classes_[np.argmax(proba, axis=1)]
        cm_path = os.path.join(emission_dir, "class_map.json")
        if not os.path.isfile(cm_path):
            cm_path = os.path.join(os.path.dirname(emission_dir), "class_map.json")
        with open(cm_path) as f:
            class_map = json.load(f)
    elif kind == "ppca_mixture":
        from src.models.ppca_classifier import PPCAMixtureClassifier
        clf = PPCAMixtureClassifier.load(emission_dir)
        proba = clf.predict_proba(X)
        preds = clf.classes_[np.argmax(proba, axis=1)]
        cm_path = os.path.join(emission_dir, "class_map.json")
        if not os.path.isfile(cm_path):
            cm_path = os.path.join(os.path.dirname(emission_dir), "class_map.json")
        with open(cm_path) as f:
            class_map = json.load(f)
    else:
        with open(os.path.join(emission_dir, "model.pkl"), "rb") as f:
            clf = pickle.load(f)
        with open(os.path.join(emission_dir, "class_map.json")) as f:
            class_map = json.load(f)
        scaler_path = os.path.join(emission_dir, "scaler.npz")
        if os.path.isfile(scaler_path):
            sd = np.load(scaler_path)
            X = (X - sd["mean"]) / sd["scale"]
        proba = clf.predict_proba(X)
        preds = clf.classes_[np.argmax(proba, axis=1)]

    idx_to_letter = {int(v): k.upper() for k, v in class_map.items()}
    pred_letters = np.array([idx_to_letter.get(int(p), "?") for p in preds])
    top_proba = proba[np.arange(len(preds)), np.argmax(proba, axis=1)]
    return pred_letters, top_proba, idx_to_letter


def predict_with_ctc(checkpoint, X, device):
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
        log_probs = model(x, lengths).squeeze(1).cpu().numpy()  # (T, 27)
    probs = np.exp(log_probs)
    pred_idx = probs.argmax(axis=1)
    pred_letters = []
    pred_probs = []
    for t in range(len(pred_idx)):
        idx = pred_idx[t]
        pred_probs.append(float(probs[t, idx]))
        if idx == 0:
            pred_letters.append("_")  # CTC blank
        else:
            pred_letters.append(_LETTERS[idx - 1])
    return np.array(pred_letters), np.array(pred_probs)


def find_holds(pred_letters, min_hold_frames, smooth_window=11):
    T = len(pred_letters)
    half = smooth_window // 2
    smoothed = []
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        most_common = Counter(pred_letters[lo:hi]).most_common(1)[0][0]
        smoothed.append(most_common)
    sm = np.array(smoothed)

    runs = []
    prev = None
    start = 0
    for t in range(T):
        if sm[t] != prev:
            if prev is not None:
                runs.append((prev, start, t))
            prev = sm[t]
            start = t
    runs.append((prev, start, T))
    return [(l, s, e) for l, s, e in runs if (e - s) >= min_hold_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--rf-dir", default="experiments/rf_balanced")
    parser.add_argument("--mppca-dir", default="experiments/mppca_balanced/mixture")
    parser.add_argument("--ppca-dir", default="experiments/ppca_balanced/mixture")
    parser.add_argument("--ctc-checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--tile-size", type=float, default=2.5)
    parser.add_argument("--min-hold-seconds", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Extracting frames + landmarks from {args.input}")
    frames, X, fps = extract_frames_and_landmarks(args.input)
    print(f"  T={len(X)} frames @ {fps:.1f} fps")

    print(f"Running RF...")
    rf_pred, rf_prob, _ = predict_proba_with_model(args.rf_dir, X, kind="sklearn")

    print(f"Running MPPCA...")
    mp_pred, mp_prob, _ = predict_proba_with_model(args.mppca_dir, X, kind="mppca")

    print(f"Running PPCA-Mixture...")
    pp_pred, pp_prob, _ = predict_proba_with_model(args.ppca_dir, X, kind="ppca_mixture")

    print(f"Running CTC v2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctc_pred, ctc_prob = predict_with_ctc(args.ctc_checkpoint, X, device)

    # Use RF holds as anchor (it's the strongest per-frame model)
    min_hold_frames = max(1, int(args.min_hold_seconds * fps))
    holds = find_holds(rf_pred, min_hold_frames, smooth_window=11)
    print(f"\n{len(holds)} RF holds detected")

    # Pick the middle frame of each hold as the representative
    n = len(holds)
    cols = args.cols
    rows = (n + cols - 1) // cols

    tw = args.tile_size
    th = tw * 1.4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * tw, rows * th))
    if rows == 1:
        axes = np.array([axes])

    for i, (rf_letter, s, e) in enumerate(holds):
        ax = axes[i // cols, i % cols]
        mid = (s + e) // 2
        if mid >= len(frames):
            mid = len(frames) - 1
        img = cv2.cvtColor(frames[mid], cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        ctc_letter_str = ctc_pred[mid] if ctc_pred[mid] != "_" else "BLANK"
        title = (
            f"#{i+1}  frame {mid}\n"
            f"RF:    {rf_letter}      ({rf_prob[mid]:.2f})\n"
            f"MPPCA: {mp_pred[mid]}      ({mp_prob[mid]:.2f})\n"
            f"PPCA:  {pp_pred[mid]}      ({pp_prob[mid]:.2f})\n"
            f"CTC:   {ctc_letter_str}  ({ctc_prob[mid]:.2f})"
        )
        ax.set_title(title, fontsize=9, family="monospace", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    out = args.output or os.path.join(
        "segments",
        os.path.splitext(os.path.basename(args.input))[0],
        "model_comparison_grid.png",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
