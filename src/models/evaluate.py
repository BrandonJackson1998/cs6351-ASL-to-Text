"""
Evaluation script for all models.
Computes accuracy, F1, confusion matrix, and classification reports.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.mlp import FingerspellingMLP
from src.preprocessing.dataset import ASLAlphabetDataset


def _load_ppca_classifier(checkpoint_dir):
    cfg_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"PPCA config.json not found in {checkpoint_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg["type"] == "ppca_logistic":
        from src.models.ppca_classifier import PPCALogisticClassifier
        return PPCALogisticClassifier.load(checkpoint_dir)
    if cfg["type"] == "ppca_mixture":
        from src.models.ppca_classifier import PPCAMixtureClassifier
        return PPCAMixtureClassifier.load(checkpoint_dir)
    raise ValueError(f"Unknown PPCA classifier type: {cfg['type']}")


def evaluate(args):
    data_dir = args.data_dir or "data/asl_alphabet"
    landmarks_dir = os.path.join(data_dir, "landmarks")
    split = args.split

    if args.model == "mlp":
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}'")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(args.checkpoint, map_location=device)
        num_classes = ckpt.get("num_classes", 26)
        model = FingerspellingMLP(input_dim=63, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        class_map = ckpt.get("class_map", {})
        idx_to_class = {v: k for k, v in class_map.items()}

        dataset = ASLAlphabetDataset(landmarks_dir, split=split)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                preds = model(x).argmax(1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(y.numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

    elif args.model == "ppca":
        if not os.path.isdir(args.checkpoint):
            raise NotADirectoryError(
                f"PPCA checkpoint must be a directory: '{args.checkpoint}'"
            )
        clf = _load_ppca_classifier(args.checkpoint)
        dataset = ASLAlphabetDataset(landmarks_dir, split=split)
        X, y = dataset.get_arrays()
        all_preds = clf.predict(X)
        all_labels = y
        idx_to_class = dataset.idx_to_class
        num_classes = dataset.num_classes

    else:
        raise NotImplementedError(f"Evaluation for model '{args.model}' not implemented")

    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]
    acc = (all_preds == all_labels).mean()
    print(f"\n{split} accuracy: {acc:.4f}  ({int(acc * len(all_labels))}/{len(all_labels)})")
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
    )

    if args.confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix ({args.model}, {split})")
        plt.tight_layout()
        plt.savefig(args.confusion_matrix, dpi=150)
        print(f"Confusion matrix saved to {args.confusion_matrix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", choices=["mlp", "lstm", "ppca"], required=True)
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .pt file (mlp/lstm) or directory (ppca)",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--confusion-matrix", type=str, default=None,
                        help="Optional path to save confusion matrix PNG")
    args = parser.parse_args()
    evaluate(args)
