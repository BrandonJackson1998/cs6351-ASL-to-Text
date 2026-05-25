"""Build a per-letter, class-balanced training pool by combining Kaggle ASL
Alphabet landmarks with FSWild CTC-forced-aligned frames.

For each letter A-Z:
  1. Pool all available landmarks (Kaggle + FSWild)
  2. If pool size > target_per_letter, randomly subsample to target
  3. If pool size < target_per_letter, oversample with landmark-level augmentation

Output layout (matches Kaggle alphabet so existing trainers/datasets work):
    data/balanced_letter_frames/landmarks/A.npy
    data/balanced_letter_frames/landmarks/B.npy
    ...
    data/balanced_letter_frames/class_map.json
"""

import argparse
import json
import os
import string

import numpy as np

from src.preprocessing.augment import jitter, rotate_3d


_LETTERS = list(string.ascii_uppercase)


def _load_letter_pools(landmarks_dir):
    """Return {LETTER: (N, 63) np.float32} from a Kaggle-style layout."""
    class_map_path = os.path.join(os.path.dirname(landmarks_dir), "class_map.json")
    if not os.path.isfile(class_map_path):
        return {}
    with open(class_map_path) as f:
        raw = json.load(f)

    pools = {}
    for name in raw:
        letter = name.upper()
        if letter not in _LETTERS:
            continue
        npy_path = os.path.join(landmarks_dir, f"{name}.npy")
        if not os.path.isfile(npy_path):
            continue
        arr = np.load(npy_path).astype(np.float32)
        pools[letter] = arr
    return pools


def _augmented_oversample(pool, target, rng, sigma=0.015, max_rot=10.0):
    """Oversample `pool` to `target` rows by random landmark-level augmentation."""
    out = [pool]
    needed = target - len(pool)
    while needed > 0:
        # Pick a random base sample, augment, append
        idx = rng.integers(0, len(pool))
        x = pool[idx]
        # random pick of jitter / rotate
        choice = rng.random()
        if choice < 0.5:
            x_aug = jitter(x, sigma=sigma, rng=rng)
        else:
            x_aug = rotate_3d(x, max_deg=max_rot, rng=rng)
        out.append(x_aug.reshape(1, -1))
        needed -= 1
    return np.concatenate(out, axis=0)[: target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", default="data/asl_alphabet/landmarks")
    parser.add_argument("--fswild-dir", default="data/chicagofswild/letter_frames/landmarks")
    parser.add_argument("--output-dir", default="data/balanced_letter_frames")
    parser.add_argument("--target", type=int, default=4000,
                        help="Target frames per letter (balanced).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true",
                        help="If set, only subsample / pad with copies (no augmentation).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    kaggle_pools = _load_letter_pools(args.kaggle_dir)
    fswild_pools = _load_letter_pools(args.fswild_dir)
    print(f"Kaggle letters: {len(kaggle_pools)}  FSWild letters: {len(fswild_pools)}")

    out_landmarks = os.path.join(args.output_dir, "landmarks")
    os.makedirs(out_landmarks, exist_ok=True)
    class_map = {c: i for i, c in enumerate(_LETTERS)}
    with open(os.path.join(args.output_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"\nBalancing each letter to target={args.target} frames")
    print(f"{'Letter':>6}  {'Kaggle':>8}  {'FSWild':>8}  {'Pooled':>8}  {'Final':>8}  {'Source':>15}")
    print("-" * 65)
    counts = {}
    for letter in _LETTERS:
        k = kaggle_pools.get(letter, np.zeros((0, 63), dtype=np.float32))
        f_ = fswild_pools.get(letter, np.zeros((0, 63), dtype=np.float32))
        pool = np.concatenate([k, f_], axis=0) if (len(k) + len(f_)) > 0 else np.zeros((0, 63), dtype=np.float32)
        if len(pool) == 0:
            print(f"  {letter}: NO DATA, skipping")
            counts[letter] = 0
            continue
        if len(pool) >= args.target:
            idx = rng.choice(len(pool), size=args.target, replace=False)
            final = pool[idx]
            source = "subsampled"
        elif args.no_augment:
            # Just repeat with replacement
            idx = rng.choice(len(pool), size=args.target, replace=True)
            final = pool[idx]
            source = "repeated"
        else:
            final = _augmented_oversample(pool, args.target, rng)
            source = "augmented"
        np.save(os.path.join(out_landmarks, f"{letter}.npy"), final.astype(np.float32))
        counts[letter] = len(final)
        print(f"  {letter:>4}  {len(k):>8d}  {len(f_):>8d}  {len(pool):>8d}  {len(final):>8d}  {source:>15s}")

    total = sum(counts.values())
    print(f"\nTotal frames: {total}  (balanced to {args.target} per letter)")
    print(f"Saved to {out_landmarks}")


if __name__ == "__main__":
    main()
