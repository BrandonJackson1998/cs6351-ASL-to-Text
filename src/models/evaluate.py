"""
Evaluation script for all models.
Computes accuracy, F1, confusion matrix, and classification reports.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.models.mlp import FingerspellingMLP
from src.preprocessing.dataset import ASLAlphabetDataset


def evaluate(args):
    """Evaluate a trained model on the validation/test set."""
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: '{args.checkpoint}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.model == "mlp":
        num_classes = ckpt.get("num_classes", 29)
        model = FingerspellingMLP(input_dim=63, num_classes=num_classes).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        class_map = ckpt.get("class_map", {})
        idx_to_class = {v: k for k, v in class_map.items()}

        data_dir = args.data_dir or "data/asl_alphabet"
        landmarks_dir = os.path.join(data_dir, "landmarks")
        dataset = ASLAlphabetDataset(landmarks_dir, split="val")
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    else:
        raise NotImplementedError("LSTM evaluation not yet implemented (Phase 2)")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    class_names = [idx_to_class.get(i, str(i)) for i in range(num_classes)]
    acc = (all_preds == all_labels).mean()
    print(f"\nVal accuracy: {acc:.4f}  ({int(acc * len(all_labels))}/{len(all_labels)})")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, labels=list(range(num_classes)), target_names=class_names, zero_division=0))

    if args.confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        out = args.confusion_matrix
        plt.savefig(out, dpi=150)
        print(f"Confusion matrix saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", choices=["mlp", "lstm"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--confusion-matrix", type=str, default=None,
                        help="Optional path to save confusion matrix PNG")
    args = parser.parse_args()
    evaluate(args)
