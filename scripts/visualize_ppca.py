"""Visualize a trained PPCA classifier.

Produces three plots, saved alongside the checkpoint:
  - scree.png         : explained variance per component
  - latent_2d.png     : 2D scatter of test-split letters in PPCA space
  - confusion.png     : confusion matrix on the test split
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.ppca_classifier import PPCALogisticClassifier
from src.preprocessing.dataset import ASLAlphabetDataset


def _scree_plot(ppca, out_path):
    eigvals = (ppca.W_ ** 2).sum(axis=0) + ppca.sigma2_
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    explained = eigvals / eigvals.sum()
    cum = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(eigvals) + 1), explained, alpha=0.6, label="per component")
    ax.plot(range(1, len(eigvals) + 1), cum, "o-", color="C1", label="cumulative")
    ax.set_xlabel("Component index")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PPCA scree (n_components={ppca.n_components}, sigma^2={ppca.sigma2_:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _latent_2d(clf, X, y, idx_to_class, out_path):
    Z = clf.transform(X)
    if Z.shape[1] < 2:
        print("n_components < 2, skipping 2D plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    classes = np.unique(y)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(classes):
        mask = y == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=4, alpha=0.5,
                   color=cmap(i % 20), label=idx_to_class.get(int(c), str(c)))
    ax.set_xlabel("PPCA component 1")
    ax.set_ylabel("PPCA component 2")
    ax.set_title("Test-split landmarks in PPCA latent space")
    ax.legend(ncol=2, fontsize=7, loc="best", markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def _confusion(clf, X, y, idx_to_class, num_classes, out_path):
    preds = clf.predict(X)
    cm = confusion_matrix(y, preds, labels=list(range(num_classes)))
    names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = float((preds == y).mean())
    ax.set_title(f"Confusion matrix (test split, acc={acc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(args):
    if not os.path.isdir(args.checkpoint):
        raise NotADirectoryError(f"Expected directory: {args.checkpoint}")

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        cfg = json.load(f)
    if cfg["type"] != "ppca_logistic":
        print(f"Visualization only supports ppca_logistic; got {cfg['type']}")
        return

    clf = PPCALogisticClassifier.load(args.checkpoint)

    landmarks_dir = os.path.join(args.data_dir, "landmarks")
    test_ds = ASLAlphabetDataset(landmarks_dir, split="test")
    X, y = test_ds.get_arrays()

    _scree_plot(clf.ppca, os.path.join(args.checkpoint, "scree.png"))
    _latent_2d(clf, X, y, test_ds.idx_to_class, os.path.join(args.checkpoint, "latent_2d.png"))
    _confusion(clf, X, y, test_ds.idx_to_class, test_ds.num_classes,
               os.path.join(args.checkpoint, "confusion.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to PPCA experiment directory")
    parser.add_argument("--data-dir", type=str, default="data/asl_alphabet")
    args = parser.parse_args()
    main(args)
