"""ChicagoFSWild dataset adapter.

Reads ChicagoFSWild.csv + extracted frame sequences, runs MediaPipe Hands
on each sequence, and saves per-sequence (T, 63) landmark arrays + labels.

Usage:
    python -m src.preprocessing.chicagofswild --extract-landmarks
    python -m src.preprocessing.chicagofswild --extract-landmarks --partition train

Output structure (under data/chicagofswild/landmarks/):
    train.npz   — sequences (object array of (T, 63) float32) + labels (str)
    dev.npz
    test.npz
"""

import argparse
import csv
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import cv2
from tqdm import tqdm

from src.preprocessing.landmark_extractor import (
    extract_hand_landmarks,
    _make_landmarker,
    normalize_landmarks,
)


_ALPHABET = set("abcdefghijklmnopqrstuvwxyz")


def load_metadata(csv_path: str):
    """Yield dict rows from ChicagoFSWild.csv."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_frames(frames_dir: str) -> List[str]:
    """Return sorted list of frame paths in a sequence directory."""
    if not os.path.isdir(frames_dir):
        return []
    files = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    return files


def extract_sequence_landmarks(
    frame_paths: List[str],
    landmarker,
    fill_with_last: bool = True,
) -> Optional[np.ndarray]:
    """Run MediaPipe per-frame, return (T, 63) array. Missing frames filled
    with last seen landmark (or zeros if none seen yet).
    """
    out = []
    last = np.zeros(63, dtype=np.float32)
    have_any = False
    for p in frame_paths:
        lm = extract_hand_landmarks(p, landmarker=landmarker)
        if lm is not None:
            last = lm
            have_any = True
            out.append(lm.copy())
        elif fill_with_last and have_any:
            out.append(last.copy())
        # If we've never seen a hand yet, skip the frame entirely
    if not out:
        return None
    return np.stack(out)


def clean_label(label: str) -> str:
    """Map FSWild labels to A-Z only.

    The processed label uses lowercase letters and '<sp>' for space. We drop
    the space marker (CTC will learn implicit boundaries) and uppercase.
    """
    if label is None:
        return ""
    label = label.replace("<sp>", "").replace(" ", "")
    out = "".join(c.upper() for c in label if c.lower() in _ALPHABET)
    return out


def build_landmark_dataset(
    csv_path: str,
    frames_root: str,
    output_dir: str,
    partition: str = "all",
    limit: Optional[int] = None,
):
    """Process FSWild sequences in a partition and save landmark arrays."""
    os.makedirs(output_dir, exist_ok=True)

    rows = list(load_metadata(csv_path))
    print(f"Loaded {len(rows)} rows from {csv_path}")

    if partition == "all":
        partitions = ["train", "dev", "test"]
    else:
        partitions = [partition]

    landmarker = _make_landmarker()

    for part in partitions:
        part_rows = [r for r in rows if r.get("partition") == part]
        if limit:
            part_rows = part_rows[:limit]
        print(f"\n=== {part}: {len(part_rows)} sequences ===")

        sequences = []
        labels = []
        skipped_no_frames = 0
        skipped_no_label = 0
        skipped_no_landmarks = 0

        for row in tqdm(part_rows, desc=part):
            label = clean_label(row.get("label_proc", ""))
            if not label:
                skipped_no_label += 1
                continue

            seq_dir = os.path.join(frames_root, row["filename"])
            frames = load_frames(seq_dir)
            if not frames:
                skipped_no_frames += 1
                continue

            seq = extract_sequence_landmarks(frames, landmarker)
            if seq is None or len(seq) < 2:
                skipped_no_landmarks += 1
                continue

            sequences.append(seq.astype(np.float32))
            labels.append(label)

        out_path = os.path.join(output_dir, f"{part}.npz")
        np.savez(
            out_path,
            sequences=np.array(sequences, dtype=object),
            labels=np.array(labels),
        )
        print(
            f"Saved {len(sequences)} sequences to {out_path}\n"
            f"  skipped: no_frames={skipped_no_frames}  "
            f"no_label={skipped_no_label}  no_landmarks={skipped_no_landmarks}"
        )
        if sequences:
            seq_lens = [len(s) for s in sequences]
            label_lens = [len(l) for l in labels]
            print(
                f"  seq length: min={min(seq_lens)}  max={max(seq_lens)}  "
                f"mean={np.mean(seq_lens):.1f}"
            )
            print(
                f"  label length: min={min(label_lens)}  max={max(label_lens)}  "
                f"mean={np.mean(label_lens):.1f}"
            )

    landmarker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/chicagofswild/ChicagoFSWild.csv")
    parser.add_argument("--frames-root", default="data/chicagofswild/ChicagoFSWild-Frames")
    parser.add_argument("--output-dir", default="data/chicagofswild/landmarks")
    parser.add_argument("--partition", choices=["all", "train", "dev", "test"], default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N sequences (for testing)")
    args = parser.parse_args()
    build_landmark_dataset(
        csv_path=args.csv,
        frames_root=args.frames_root,
        output_dir=args.output_dir,
        partition=args.partition,
        limit=args.limit,
    )
