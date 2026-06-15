"""Train an RBF-kernel SVM letter classifier on FSWild forced-aligned frames.

Course reference: Mod8-NonlinSVM.pdf. Hyperparameters:
  - C: soft-margin penalty, log-grid 0.1..100
  - gamma: RBF bandwidth, log-grid around 1/(p * Var(x))
  - probability=True so we can use predict_proba in fusion

Same data layout as the PPCA refit and RF (ASLAlphabetDataset on
data/chicagofswild/letter_frames/landmarks). Same 60/20/20 split, so all
three per-frame models are directly comparable.

Note: kernel SVM scales superlinearly with n_train. With ~34k training
samples this can take 10-30 minutes per (C, gamma) combination on CPU.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.dataset import ASLAlphabetDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/chicagofswild/letter_frames")
    parser.add_argument("--Cs", type=float, nargs="+", default=[1.0, 10.0],
                        help="C values (soft-margin penalty) to sweep on val")
    parser.add_argument("--gammas", nargs="+", default=["scale"],
                        help="gamma values: 'scale' (=1/(p*Var(X))), 'auto' (=1/p), or a float")
    parser.add_argument("--max-train", type=int, default=20000,
                        help="Subsample training set to this size (kernel SVM scales O(n^2-n^3))")
    parser.add_argument("--seed", type=int, default=42)
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

    # Subsample training if too large (kernel SVM cost)
    if args.max_train and len(X_train) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_train), size=args.max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"Subsampled train to {len(X_train)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Parse gamma values
    parsed_gammas = []
    for g in args.gammas:
        try:
            parsed_gammas.append(float(g))
        except ValueError:
            parsed_gammas.append(g)

    print(f"\nSweeping C={args.Cs}, gamma={parsed_gammas}")
    sweep = []
    best = {"val_acc": -1.0, "model": None, "C": None, "gamma": None}
    for C in args.Cs:
        for gamma in parsed_gammas:
            t0 = time.time()
            clf = SVC(C=C, gamma=gamma, kernel="rbf", probability=True,
                      cache_size=512, decision_function_shape="ovr",
                      random_state=args.seed)
            clf.fit(X_train_s, y_train)
            train_time = time.time() - t0
            train_acc = clf.score(X_train_s, y_train)
            val_acc = clf.score(X_val_s, y_val)
            print(f"  C={C}  gamma={gamma}  train={train_acc:.4f}  val={val_acc:.4f}  "
                  f"({train_time:.1f}s, {clf.support_.shape[0]} SVs)")
            sweep.append({
                "C": float(C), "gamma": str(gamma),
                "train_acc": float(train_acc), "val_acc": float(val_acc),
                "n_support": int(clf.support_.shape[0]),
                "train_time_s": float(train_time),
            })
            if val_acc > best["val_acc"]:
                best = {"val_acc": val_acc, "model": clf, "C": C, "gamma": gamma}

    print(f"\nBest: C={best['C']}  gamma={best['gamma']}  val_acc={best['val_acc']:.4f}")
    test_acc = best["model"].score(X_test_s, y_test)
    print(f"Test acc: {test_acc:.4f}")

    test_preds = best["model"].predict(X_test_s)
    class_names = [train_ds.idx_to_class[i] for i in range(train_ds.num_classes)]
    print("\nClassification report (test):")
    print(classification_report(
        y_test, test_preds,
        labels=list(range(train_ds.num_classes)),
        target_names=class_names, zero_division=0,
    ))

    cm = confusion_matrix(y_test, test_preds, labels=list(range(train_ds.num_classes)))
    np.fill_diagonal(cm, 0)
    confusions = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    print("Top-10 confusions:")
    for count, true, pred in confusions[:10]:
        print(f"  {true} → {pred}  ({count})")

    exp_name = args.exp_name or f"svm_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    np.savez(
        os.path.join(exp_dir, "scaler.npz"),
        mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_,
        n_features=scaler.n_features_in_,
    )
    with open(os.path.join(exp_dir, "class_map.json"), "w") as f:
        json.dump(train_ds.class_map, f, indent=2)
    metrics = {
        "sweep": sweep,
        "best_C": float(best["C"]), "best_gamma": str(best["gamma"]),
        "best_val_acc": float(best["val_acc"]),
        "test_acc": float(test_acc),
        "data_dir": args.data_dir,
        "max_train": int(args.max_train),
        "top_confusions": [
            {"true": t, "pred": p, "count": int(c)} for c, t, p in confusions[:20]
        ],
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved SVM to {exp_dir}")


if __name__ == "__main__":
    main()
