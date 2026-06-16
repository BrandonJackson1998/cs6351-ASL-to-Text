#!/usr/bin/env python3
"""
Evaluate RF vs PPCA vs Ensemble on test data.

Compares accuracy on held-out test frames.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ppca_classifier import PPCAMixtureClassifier
from src.preprocessing.dataset import ASLAlphabetDataset


def load_test_data(data_dir: Path):
    """Load test split from balanced dataset."""
    landmarks_dir = data_dir / "landmarks"
    test_ds = ASLAlphabetDataset(str(landmarks_dir), split="test")
    X_test, y_test = test_ds.get_arrays()
    return X_test, y_test


def ensemble_predict(rf_model, ppca_model, X, rf_weight=0.6):
    """Ensemble predictions."""
    rf_proba = rf_model.predict_proba(X)
    ppca_proba = ppca_model.predict_proba(X)

    ensemble_proba = rf_weight * rf_proba + (1 - rf_weight) * ppca_proba
    predictions = ensemble_proba.argmax(axis=1)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate RF vs PPCA vs Ensemble")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/balanced_letter_frames"),
        help="Data directory with test.npy (default: data/balanced_letter_frames)"
    )

    parser.add_argument(
        "--rf-model",
        type=Path,
        default=Path("experiments/rf_balanced/model.pkl"),
        help="RF model path"
    )

    parser.add_argument(
        "--ppca-model",
        type=Path,
        default=Path("experiments/ppca_20260615_212735"),
        help="PPCA model directory"
    )

    parser.add_argument(
        "--rf-weight",
        type=float,
        default=0.6,
        help="Ensemble RF weight (default: 0.6)"
    )

    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print confusion matrix"
    )

    args = parser.parse_args()

    # Load data
    print("Loading test data...")
    X_test, y_test = load_test_data(args.data_dir)
    print(f"Test samples: {len(X_test)}")

    # Load class map
    class_map_path = args.rf_model.parent / "class_map.json"
    with open(class_map_path) as f:
        letter_to_idx = json.load(f)
        idx_to_letter = {v: k for k, v in letter_to_idx.items()}

    # Load models
    print("\nLoading models...")
    with open(args.rf_model, "rb") as f:
        rf_model = pickle.load(f)

    ppca_model = PPCAMixtureClassifier.load(str(args.ppca_model / "mixture"))

    # Evaluate RF
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

    # Evaluate PPCA
    print("\n" + "=" * 60)
    print("PPCA MIXTURE")
    print("=" * 60)
    ppca_pred = ppca_model.predict(X_test)
    ppca_acc = accuracy_score(y_test, ppca_pred)
    print(f"Accuracy: {ppca_acc:.4f} ({ppca_acc*100:.2f}%)")

    # Evaluate Ensemble
    print("\n" + "=" * 60)
    print(f"ENSEMBLE (RF={args.rf_weight:.1f}, PPCA={1-args.rf_weight:.1f})")
    print("=" * 60)
    ensemble_pred = ensemble_predict(rf_model, ppca_model, X_test, args.rf_weight)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"RF:        {rf_acc*100:6.2f}%")
    print(f"PPCA:      {ppca_acc*100:6.2f}%")
    print(f"Ensemble:  {ensemble_acc*100:6.2f}%")
    print()
    print(f"Improvement over RF:   {(ensemble_acc - rf_acc)*100:+.2f}%")
    print(f"Improvement over PPCA: {(ensemble_acc - ppca_acc)*100:+.2f}%")

    # Confusion matrix
    if args.confusion_matrix:
        print("\n" + "=" * 60)
        print("ENSEMBLE CONFUSION MATRIX")
        print("=" * 60)
        cm = confusion_matrix(y_test, ensemble_pred)
        letters = sorted(idx_to_letter.keys())

        # Print header
        print("     ", end="")
        for i in letters:
            print(f"{idx_to_letter[i]:>4}", end="")
        print()

        # Print rows
        for i in letters:
            print(f"{idx_to_letter[i]:>4} ", end="")
            for j in letters:
                val = cm[i, j]
                if val > 0:
                    print(f"{val:>4}", end="")
                else:
                    print("   .", end="")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
