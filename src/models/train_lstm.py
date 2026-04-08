"""
Training script for the LSTM word recognition model (Phase 2).
"""

import argparse


def train(args):
    """Train the LSTM sequence model."""
    raise NotImplementedError("Phase 2 - implement LSTM training loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM word sign classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/wlasl")
    parser.add_argument("--num-words", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()
    train(args)
