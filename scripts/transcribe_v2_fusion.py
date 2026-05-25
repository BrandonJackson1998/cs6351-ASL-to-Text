"""Transcribe a video using CTC v2 + a per-frame emission model fusion.

Supports either Random Forest (sklearn .pkl) or RBF-SVM (sklearn .pkl + scaler).
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import mediapipe as mp

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.models.decode_ctc import (
    greedy_decode, beam_search_decode, beam_search_decode_lm, fuse_with_ppca,
)
from src.models.char_lm import build_default_lm
from src.preprocessing.combined_dataset import NUM_CLASSES
from src.preprocessing.landmark_extractor import _make_landmarker, normalize_landmarks


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EPS = 1e-8


def extract_landmark_stream(video_path, fill_with_last=True):
    landmarker = _make_landmarker()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    out = []
    last_seen = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            raw = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()
            last_seen = normalize_landmarks(raw)
            out.append(last_seen.copy())
        elif fill_with_last and last_seen is not None:
            out.append(last_seen.copy())
    cap.release()
    landmarker.close()
    return (np.stack(out) if out else None), fps


def emission_log_probs(emission_dir, seq):
    """Return (T, 26) log-probs from RF or SVM."""
    with open(os.path.join(emission_dir, "model.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(emission_dir, "class_map.json")) as f:
        class_map = json.load(f)
    scaler_path = os.path.join(emission_dir, "scaler.npz")

    X = seq.astype(np.float32)
    if os.path.isfile(scaler_path):
        sd = np.load(scaler_path)
        X = (X - sd["mean"]) / sd["scale"]

    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    out = np.full((seq.shape[0], 26), -10.0, dtype=np.float64)
    for class_name, class_idx in class_map.items():
        letter = class_name.upper()
        if letter not in _LETTERS or class_idx not in classes:
            continue
        col = _LETTERS.index(letter)
        cc = classes.index(class_idx)
        out[:, col] = np.log(proba[:, cc] + _EPS)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--emission-dir", required=True,
                        help="Directory with model.pkl + class_map.json (and scaler.npz for SVM)")
    parser.add_argument("--lam", type=float, default=0.25,
                        help="Fusion weight (0 = pure CTC, >0 = blend in emission model)")
    parser.add_argument("--lm-alpha", type=float, default=0.25)
    parser.add_argument("--lm-beta", type=float, default=0.0)
    parser.add_argument("--lm-n", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CTC v2 from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = CTCLSTMv2(
        input_dim=63,
        hidden_dim=train_args.get("hidden_dim", 256),
        num_layers=train_args.get("num_layers", 3),
        dropout=train_args.get("dropout", 0.3),
        num_classes=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Extracting landmarks from {args.input}")
    seq, fps = extract_landmark_stream(args.input)
    if seq is None:
        print("ERROR: no landmarks", file=sys.stderr)
        sys.exit(1)
    print(f"  fps={fps:.1f}  T={len(seq)}")

    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
    lengths = torch.tensor([len(seq)], dtype=torch.long)
    with torch.no_grad():
        log_probs = model(x, lengths)
    ctc_lp = log_probs.squeeze(1).cpu().numpy()

    print(f"\nLoading emission model from {args.emission_dir}")
    emit_lp = emission_log_probs(args.emission_dir, seq)

    print("Building character LM...")
    char_lm = build_default_lm(n=args.lm_n)

    print("\n=== Decoded outputs ===")
    print(f"-- Pure CTC (lam=0) --")
    g0 = greedy_decode(ctc_lp)
    print(f"  Greedy:        {g0}")
    b0 = beam_search_decode_lm(ctc_lp, char_lm,
                                alpha=args.lm_alpha, beta=args.lm_beta, beam_size=args.beam_size)
    print(f"  Beam + LM:     {b0[0]}")

    fused = fuse_with_ppca(ctc_lp, emit_lp, args.lam)
    print(f"\n-- Fused (lam={args.lam}) --")
    g1 = greedy_decode(fused)
    print(f"  Greedy:        {g1}")
    b1 = beam_search_decode_lm(fused, char_lm,
                                alpha=args.lm_alpha, beta=args.lm_beta, beam_size=args.beam_size)
    print(f"  Beam + LM:     {b1[0]}")
    if len(b1) > 1:
        print("  Top-5:")
        for i, h in enumerate(b1[:5], 1):
            print(f"    {i}. {h}")


if __name__ == "__main__":
    main()
