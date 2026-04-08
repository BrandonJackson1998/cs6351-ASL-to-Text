"""
Evaluation script for all models.
Computes accuracy, F1, confusion matrix, and classification reports.
"""

import argparse


def evaluate(args):
    """Evaluate a trained model on the test set."""
    raise NotImplementedError("Implement evaluation pipeline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", choices=["mlp", "lstm"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()
    evaluate(args)
