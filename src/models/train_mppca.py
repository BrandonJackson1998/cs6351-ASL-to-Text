"""Train MPPCAClassifier (K Gaussians per letter) on FSWild forced-aligned frames.

Sweeps K (components per letter) and q (latent dim) on val. Test once.
"""

import argparse
import json
import os
import time

import numpy as np

from src.models.mppca import MPPCAClassifier
from src.preprocessing.dataset import ASLAlphabetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/chicagofswild/letter_frames")
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 5],
                        help="components per letter to sweep on val")
    parser.add_argument("--qs", type=int, nargs="+", default=[12],
                        help="latent dimensions to sweep on val")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()

    landmarks_dir = os.path.join(args.data_dir, "landmarks")
    train_ds = ASLAlphabetDataset(landmarks_dir, split="train", augment=False)
    val_ds = ASLAlphabetDataset(landmarks_dir, split="val")
    test_ds = ASLAlphabetDataset(landmarks_dir, split="test")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    X_train, y_train = train_ds.get_arrays()
    X_val, y_val = val_ds.get_arrays()
    X_test, y_test = test_ds.get_arrays()

    sweep = []
    best = {"val_acc": -1.0, "model": None, "K": None, "q": None}
    for K in args.ks:
        for q in args.qs:
            print(f"\nFitting MPPCA (K={K}, q={q})")
            t0 = time.time()
            clf = MPPCAClassifier(
                n_components_per_class=K, latent_dim=q,
                max_iter=args.max_iter, seed=42,
            )
            clf.fit(X_train, y_train)
            train_time = time.time() - t0
            train_acc = clf.score(X_train, y_train)
            val_acc = clf.score(X_val, y_val)
            print(f"  K={K} q={q}  train={train_acc:.4f}  val={val_acc:.4f}  ({train_time:.1f}s)")
            sweep.append({
                "K": K, "q": q, "train_acc": float(train_acc),
                "val_acc": float(val_acc), "train_time_s": float(train_time),
            })
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "model": clf, "K": K, "q": q}

    print(f"\nBest K={best['K']} q={best['q']}  val={best['val_acc']:.4f}")
    test_acc = best["model"].score(X_test, y_test)
    print(f"Test acc: {test_acc:.4f}")

    exp_name = args.exp_name or f"mppca_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    best["model"].save(os.path.join(exp_dir, "mixture"))
    with open(os.path.join(exp_dir, "class_map.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=2)
    with open(os.path.join(exp_dir, "mixture", "class_map.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=2)
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump({
            "sweep": sweep,
            "best_K": best["K"], "best_q": best["q"],
            "best_val_acc": best["val_acc"],
            "test_acc": test_acc,
            "data_dir": args.data_dir,
        }, f, indent=2)
    print(f"\nSaved MPPCA to {exp_dir}")


if __name__ == "__main__":
    main()
