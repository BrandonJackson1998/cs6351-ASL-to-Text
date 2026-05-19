"""
Training script for the MLP fingerspelling classifier (Phase 1).
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.mlp import FingerspellingMLP
from src.preprocessing.dataset import ASLAlphabetDataset


def train(args):
    """Train the MLP baseline model."""
    landmarks_dir = os.path.join(args.data_dir, "landmarks")

    train_ds = ASLAlphabetDataset(landmarks_dir, split="train")
    val_ds = ASLAlphabetDataset(landmarks_dir, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}  |  Classes: {train_ds.num_classes}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FingerspellingMLP(input_dim=63, num_classes=train_ds.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    exp_name = args.exp_name or f"mlp_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Keep a symlink experiments/latest → most recent run
    latest = os.path.join("experiments", "latest")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(exp_dir), latest)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()

        t_loss = train_loss / len(train_ds)
        t_acc = train_correct / len(train_ds)
        v_loss = val_loss / len(val_ds)
        v_acc = val_correct / len(val_ds)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train loss {t_loss:.4f}  acc {t_acc:.4f}  |  "
            f"val loss {v_loss:.4f}  acc {v_acc:.4f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_path = os.path.join(exp_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": v_acc,
                    "num_classes": train_ds.num_classes,
                    "class_map": train_ds.class_map,
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (val acc {v_acc:.4f}) to {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {os.path.join(exp_dir, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP fingerspelling classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/asl_alphabet")
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()
    train(args)
