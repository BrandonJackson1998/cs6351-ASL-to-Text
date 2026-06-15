"""Extract per-letter FSWild frames via CTC forced alignment.

For each FSWild train sequence:
  1. Run CTC v2 → (T, 27) log-probs
  2. Forced-align against the ground-truth label
  3. For each frame assigned to a letter, append the (63,) landmark vector
     to that letter's bucket

Output:
    data/chicagofswild/letter_frames/A.npy   (N_A, 63)
    data/chicagofswild/letter_frames/B.npy   ...
    data/chicagofswild/letter_frames/class_map.json   (matches Kaggle layout)
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.models.ctc_forced_align import forced_align, BLANK_CHAR
from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.preprocessing.combined_dataset import NUM_CLASSES


_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--fswild-train", default="data/chicagofswild/landmarks/train.npz")
    parser.add_argument("--output-dir", default="data/chicagofswild/letter_frames")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N sequences (for testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CTC v2 from {args.ctc_checkpoint}")
    ckpt = torch.load(args.ctc_checkpoint, map_location=device, weights_only=False)
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

    print(f"Loading FSWild train from {args.fswild_train}")
    data = np.load(args.fswild_train, allow_pickle=True)
    sequences = data["sequences"]
    labels = data["labels"]
    n = len(labels)
    if args.limit:
        sequences = sequences[: args.limit]
        labels = labels[: args.limit]
        n = len(labels)
    print(f"  {n} sequences")

    by_letter = {c: [] for c in _LETTERS}
    n_aligned = 0
    n_failed = 0

    with torch.no_grad():
        for i in tqdm(range(n), desc="forced-align"):
            seq = sequences[i].astype(np.float32)
            label = str(labels[i])
            if not any(c.upper() in _LETTERS for c in label) or len(seq) < 1:
                n_failed += 1
                continue
            x = torch.from_numpy(seq).unsqueeze(0).to(device)
            lengths = torch.tensor([len(seq)], dtype=torch.long)
            log_probs = model(x, lengths).squeeze(1).cpu().numpy()  # (T, V)
            assignment = forced_align(log_probs, label)
            if assignment is None:
                n_failed += 1
                continue
            n_aligned += 1
            for t, letter in enumerate(assignment):
                if letter == BLANK_CHAR:
                    continue
                by_letter[letter].append(seq[t])

    print(f"\nAligned: {n_aligned}/{n}  Failed: {n_failed}")

    os.makedirs(args.output_dir, exist_ok=True)
    class_map = {c: i for i, c in enumerate(_LETTERS)}
    with open(os.path.join(os.path.dirname(args.output_dir), "letter_frames_class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)
    # Mirror layout that PPCAMixtureClassifier.fit + dataset code expects:
    #   <parent>/class_map.json + <parent>/landmarks/<NAME>.npy
    out_landmarks = os.path.join(args.output_dir, "landmarks")
    os.makedirs(out_landmarks, exist_ok=True)
    with open(os.path.join(args.output_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    counts = {}
    for c in _LETTERS:
        arr = np.stack(by_letter[c]) if by_letter[c] else np.zeros((0, 63), dtype=np.float32)
        np.save(os.path.join(out_landmarks, f"{c}.npy"), arr)
        counts[c] = int(arr.shape[0])
    print("\nFrames per letter:")
    for c in _LETTERS:
        print(f"  {c}: {counts[c]}")
    total = sum(counts.values())
    print(f"\nTotal frames: {total}")


if __name__ == "__main__":
    main()
