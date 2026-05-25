"""Refit PPCA-Mixture on per-letter FSWild frames extracted via CTC forced alignment.

Reads landmarks from data/chicagofswild/letter_frames/landmarks/<LETTER>.npy
(layout matches Kaggle alphabet so we can reuse ASLAlphabetDataset).

Splits 60/20/20 stratified by letter (same convention as Kaggle PPCA).
Sweeps n_components on val. Saves trained PPCAMixtureClassifier to
experiments/ppca_fswild_<timestamp>/mixture/ alongside class_map.json.
"""

import argparse
import json
import os
import time

import numpy as np

from src.models.ppca_classifier import PPCALogisticClassifier, PPCAMixtureClassifier
from src.preprocessing.dataset import ASLAlphabetDataset


def train(args):
    landmarks_dir = os.path.join(args.data_dir, "landmarks")
    if not os.path.isdir(landmarks_dir):
        raise FileNotFoundError(
            f"Expected per-letter .npy files in {landmarks_dir}. "
            "Run `make extract-fswild-letter-frames` first."
        )

    train_ds = ASLAlphabetDataset(landmarks_dir, split="train", augment=False)
    val_ds = ASLAlphabetDataset(landmarks_dir, split="val")
    test_ds = ASLAlphabetDataset(landmarks_dir, split="test")
    print(
        f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  "
        f"Classes: {train_ds.num_classes}"
    )

    X_train, y_train = train_ds.get_arrays()
    X_val, y_val = val_ds.get_arrays()
    X_test, y_test = test_ds.get_arrays()

    components_list = [int(c) for c in args.components.split(",")]
    print(f"\nSweeping n_components: {components_list}")

    sweep = []
    best = {"n_components": None, "val_acc": -1.0, "model": None}
    for n in components_list:
        clf = PPCALogisticClassifier(n_components=n, C=args.C, max_iter=args.max_iter)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        print(f"  n={n:3d}  train={train_acc:.4f}  val={val_acc:.4f}")
        sweep.append({"n_components": n, "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best["val_acc"]:
            best = {"n_components": n, "val_acc": val_acc, "model": clf}

    print(f"\nBest n_components: {best['n_components']} (val_acc={best['val_acc']:.4f})")

    # LogReg test
    test_acc = best["model"].score(X_test, y_test)
    print(f"Test acc (LogReg, FSWild-frames): {test_acc:.4f}")

    # Mixture refit at best n_components — this is the artifact we care about
    print(f"\nFitting PPCAMixtureClassifier at n_components={best['n_components']}...")
    mix = PPCAMixtureClassifier(n_components=best["n_components"])
    mix.fit(X_train, y_train)
    mix_test = mix.score(X_test, y_test)
    print(f"  mixture test acc: {mix_test:.4f}")

    exp_name = args.exp_name or f"ppca_fswild_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    best["model"].save(exp_dir)
    mix.save(os.path.join(exp_dir, "mixture"))

    # class_map needs to be at exp_dir/class_map.json AND exp_dir/mixture/class_map.json
    # for downstream tooling (evaluate_fusion uses --class-map separately).
    cm = train_ds.class_map
    with open(os.path.join(exp_dir, "class_map.json"), "w") as f:
        json.dump(cm, f, indent=2)
    with open(os.path.join(exp_dir, "mixture", "class_map.json"), "w") as f:
        json.dump(cm, f, indent=2)

    metrics = {
        "sweep": sweep,
        "best_n_components": best["n_components"],
        "best_val_acc": best["val_acc"],
        "test_acc_logistic": test_acc,
        "test_acc_mixture": mix_test,
        "C": args.C,
        "data_source": "fswild_forced_aligned",
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {exp_dir}")
    print(f"Mixture: {exp_dir}/mixture/")
    print(f"\nNext step:")
    print(f"  PYTHONPATH=. .virtual_environment/bin/python scripts/evaluate_fusion.py \\")
    print(f"    --mixture-dir {exp_dir}/mixture \\")
    print(f"    --class-map {exp_dir}/class_map.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/chicagofswild/letter_frames",
                        help="Directory containing landmarks/<LETTER>.npy + class_map.json")
    parser.add_argument("--components", default="12,20,30,40",
                        help="Comma-separated n_components to sweep on val")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()
    train(args)
