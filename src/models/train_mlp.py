"""
Training script for the MLP fingerspelling classifier (Phase 1).
"""

import argparse


def train(args):
    """Train the MLP baseline model."""
    raise NotImplementedError("Phase 1 - implement MLP training loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP fingerspelling classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/asl_alphabet")
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()
    train(args)
