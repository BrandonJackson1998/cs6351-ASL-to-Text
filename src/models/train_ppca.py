"""Train PPCA-based fingerspelling classifier (Phase 1b).

Pipeline: standardize -> PPCA -> Logistic Regression.

Sweeps n_components on the validation split, then refits with augmentation
on train and reports test-set accuracy once.
"""

import argparse
import json
import os
import time

import numpy as np

from src.models.ppca_classifier import PPCALogisticClassifier, PPCAMixtureClassifier
from src.preprocessing.dataset import ASLAlphabetDataset


def _make_train_arrays(train_ds, augment_factor=2):
    """Return (X, y) where each original sample is augmented `augment_factor` times.

    augment_factor=1 -> only one augmented copy per sample. augment_factor=2 ->
    original + augmented copy. Larger factors increase training set size.
    """
    X_orig, y_orig = train_ds.get_arrays(augment=False)
    if augment_factor <= 1:
        return train_ds.get_arrays(augment=True)
    Xs, ys = [X_orig], [y_orig]
    for _ in range(augment_factor - 1):
        X_aug, y_aug = train_ds.get_arrays(augment=True)
        Xs.append(X_aug)
        ys.append(y_aug)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def train(args):
    landmarks_dir = os.path.join(args.data_dir, "landmarks")

    train_ds = ASLAlphabetDataset(landmarks_dir, split="train", augment=True)
    val_ds = ASLAlphabetDataset(landmarks_dir, split="val")
    test_ds = ASLAlphabetDataset(landmarks_dir, split="test")
    print(
        f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  "
        f"Classes: {train_ds.num_classes}"
    )

    X_val, y_val = val_ds.get_arrays()
    X_test, y_test = test_ds.get_arrays()

    components_list = [int(c) for c in args.components.split(",")]
    print(f"\nSweeping n_components: {components_list}")

    sweep_results = []
    best = {"n_components": None, "val_acc": -1.0, "model": None}

    X_train, y_train = _make_train_arrays(train_ds, augment_factor=args.augment_factor)
    print(f"Train arrays after augmentation: {X_train.shape}")

    for n in components_list:
        clf = PPCALogisticClassifier(n_components=n, C=args.C, max_iter=args.max_iter)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        print(f"  n={n:3d}  train={train_acc:.4f}  val={val_acc:.4f}")
        sweep_results.append({"n_components": n, "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best["val_acc"]:
            best = {"n_components": n, "val_acc": val_acc, "model": clf}

    print(f"\nBest n_components: {best['n_components']} (val_acc={best['val_acc']:.4f})")

    test_acc = best["model"].score(X_test, y_test)
    print(f"Test accuracy (LogReg): {test_acc:.4f}")

    mixture_test_acc = None
    if args.mixture:
        print("\nFitting PPCAMixtureClassifier (one PPCA per class) for comparison...")
        mix = PPCAMixtureClassifier(n_components=best["n_components"])
        mix.fit(X_train, y_train)
        mix_val = mix.score(X_val, y_val)
        mixture_test_acc = mix.score(X_test, y_test)
        print(f"  Mixture val={mix_val:.4f}  test={mixture_test_acc:.4f}")

    exp_name = args.exp_name or f"ppca_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    best["model"].save(exp_dir)

    if args.mixture:
        mix_dir = os.path.join(exp_dir, "mixture")
        mix.save(mix_dir)

    metrics = {
        "sweep": sweep_results,
        "best_n_components": best["n_components"],
        "best_val_acc": best["val_acc"],
        "test_acc_logistic": test_acc,
        "test_acc_mixture": mixture_test_acc,
        "augment_factor": args.augment_factor,
        "C": args.C,
        "class_map": train_ds.class_map,
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(exp_dir, "class_map.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=2)

    latest = os.path.join("experiments", "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
    os.symlink(os.path.abspath(exp_dir), latest)

    print(f"\nSaved to {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPCA + Logistic Regression classifier")
    parser.add_argument("--data-dir", type=str, default="data/asl_alphabet")
    parser.add_argument("--components", type=str, default="8,12,16,20,30",
                        help="Comma-separated list of n_components to sweep")
    parser.add_argument("--C", type=float, default=1.0, help="LogReg inverse-regularization")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--augment-factor", type=int, default=2,
                        help="1 = augmented only; >=2 = original + (factor-1) augmented copies")
    parser.add_argument("--mixture", action="store_true",
                        help="Also fit PPCAMixtureClassifier for comparison")
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()
    train(args)
