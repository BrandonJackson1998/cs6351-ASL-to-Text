"""Train a Random Forest letter classifier on FSWild forced-aligned frames.

Same data layout as the PPCA refit (data/chicagofswild/letter_frames/), same
60/20/20 stratified split, so the two models are directly comparable on the
per-frame letter-classification task.

Course reference: Mod6-RF.pdf. Defaults follow course guidance:
  mtry ≈ sqrt(p) = sqrt(63) ≈ 8
  num.trees = 500
  min.node.size tuned on val
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.dataset import ASLAlphabetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/chicagofswild/letter_frames",
                        help="Per-letter .npy folder (Kaggle layout) — defaults to FSWild forced-aligned")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-features", default="sqrt",
                        help="mtry: 'sqrt' (=8 for p=63), 'log2', or an int")
    parser.add_argument("--min-samples-leaves", type=int, nargs="+", default=[1, 3, 5],
                        help="min_samples_leaf values to sweep on val")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()

    landmarks_dir = os.path.join(args.data_dir, "landmarks")
    train_ds = ASLAlphabetDataset(landmarks_dir, split="train", augment=False)
    val_ds = ASLAlphabetDataset(landmarks_dir, split="val")
    test_ds = ASLAlphabetDataset(landmarks_dir, split="test")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}  "
          f"Classes: {train_ds.num_classes}")

    X_train, y_train = train_ds.get_arrays()
    X_val, y_val = val_ds.get_arrays()
    X_test, y_test = test_ds.get_arrays()

    # Coerce max_features
    try:
        max_features = int(args.max_features)
    except (ValueError, TypeError):
        max_features = args.max_features

    print(f"\nSweeping min_samples_leaf={args.min_samples_leaves}, "
          f"max_features={max_features}, n_estimators={args.n_estimators}")

    sweep = []
    best = {"val_acc": -1.0, "model": None, "min_samples_leaf": None}
    for msl in args.min_samples_leaves:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_features=max_features,
            min_samples_leaf=msl,
            n_jobs=args.n_jobs,
            random_state=42,
            oob_score=True,
        )
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        oob = float(clf.oob_score_)
        print(f"  msl={msl:2d}  train={train_acc:.4f}  val={val_acc:.4f}  oob={oob:.4f}  "
              f"({train_time:.1f}s)")
        sweep.append({
            "min_samples_leaf": msl, "train_acc": train_acc, "val_acc": val_acc,
            "oob_score": oob, "train_time_s": train_time,
        })
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "model": clf, "min_samples_leaf": msl}

    print(f"\nBest min_samples_leaf={best['min_samples_leaf']}  val_acc={best['val_acc']:.4f}")

    test_acc = best["model"].score(X_test, y_test)
    print(f"Test acc: {test_acc:.4f}")

    test_preds = best["model"].predict(X_test)
    class_names = [train_ds.idx_to_class[i] for i in range(train_ds.num_classes)]
    print("\nClassification report (test):")
    print(classification_report(
        y_test, test_preds,
        labels=list(range(train_ds.num_classes)),
        target_names=class_names, zero_division=0,
    ))

    # Top confusions for the writeup
    cm = confusion_matrix(y_test, test_preds, labels=list(range(train_ds.num_classes)))
    np.fill_diagonal(cm, 0)
    confusions = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    print("Top-10 confusions (true → predicted, count):")
    for count, true, pred in confusions[:10]:
        print(f"  {true} → {pred}  ({count})")

    exp_name = args.exp_name or f"rf_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    with open(os.path.join(exp_dir, "class_map.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=2)
    metrics = {
        "sweep": sweep,
        "best_min_samples_leaf": best["min_samples_leaf"],
        "best_val_acc": best["val_acc"],
        "test_acc": test_acc,
        "data_dir": args.data_dir,
        "n_estimators": args.n_estimators,
        "max_features": str(max_features),
        "top_confusions": [
            {"true": t, "pred": p, "count": int(c)} for c, t, p in confusions[:20]
        ],
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved RF to {exp_dir}")


if __name__ == "__main__":
    main()
