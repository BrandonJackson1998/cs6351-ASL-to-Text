"""Train the v2 CTC model on the combined FSWild + Kaggle dataset.

Uses temporal + landmark augmentation, 3-layer 256-hidden Bi-LSTM, early
stopping on val LER, single test eval at the end.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.preprocessing.combined_dataset import (
    CombinedFSDataset,
    BLANK_IDX,
    NUM_CLASSES,
    collate,
)
from src.preprocessing.temporal_augment import FingerspellingAugment
from src.models.decode_ctc import greedy_decode


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model, loader, device, max_samples_to_print=8):
    model.eval()
    total_chars = 0
    total_errs = 0
    samples = []
    with torch.no_grad():
        for seqs, seq_lens, _, _, labels, sources in loader:
            seqs = seqs.to(device)
            log_probs = model(seqs, seq_lens)  # (T, B, V)
            log_probs_np = log_probs.cpu().numpy()
            for i, label in enumerate(labels):
                t = int(seq_lens[i])
                pred = greedy_decode(log_probs_np[:t, i, :])
                total_errs += levenshtein(pred, label)
                total_chars += max(1, len(label))
                if len(samples) < max_samples_to_print:
                    samples.append((label, pred, sources[i]))
    return total_errs / max(1, total_chars), samples


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug = FingerspellingAugment(seed=args.seed)
    train_ds = CombinedFSDataset("train", augment=aug)
    val_ds = CombinedFSDataset("val")
    test_ds = CombinedFSDataset("test")

    print("Train stats:", train_ds.stats())
    print("Val stats:", val_ds.stats())
    print("Test stats:", test_ds.stats())
    print(f"Device: {device}")

    # Weighted sampler so FSWild (real video, ~5k) and Kaggle (~48k) are seen
    # in equal proportion. Otherwise the LSTM never sees real temporal structure.
    n_per_source = {"fswild": 0, "kaggle": 0}
    for s in train_ds.sources:
        n_per_source[s] += 1
    src_w = {s: (1.0 / n) if n > 0 else 0.0 for s, n in n_per_source.items()}
    sample_weights = torch.tensor(
        [src_w[s] for s in train_ds.sources], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=args.steps_per_epoch * args.batch_size,
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)

    model = CTCLSTMv2(
        input_dim=63,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=NUM_CLASSES,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ctc = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    exp_name = args.exp_name or f"ctc_v2_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Saving to {exp_dir}")

    best_ler = float("inf")
    epochs_since = 0
    history = []
    best_ckpt = os.path.join(exp_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for seqs, seq_lens, targets, target_lens, _, _ in train_loader:
            seqs = seqs.to(device)
            optimizer.zero_grad()
            log_probs = model(seqs, seq_lens)
            loss = ctc(log_probs, targets, seq_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = running / max(1, n_batches)

        val_ler, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ler": val_ler})
        improved = val_ler < best_ler
        marker = "  → saved" if improved else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_LER={val_ler:.4f}{marker}", flush=True)

        if improved:
            best_ler = val_ler
            epochs_since = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_ler": val_ler,
                    "args": vars(args),
                },
                best_ckpt,
            )
        else:
            epochs_since += 1
            if args.patience > 0 and epochs_since >= args.patience:
                print(f"Early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Load best for test
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    test_ler, test_samples = evaluate(model, test_loader, device, max_samples_to_print=20)
    print(f"\nBest val LER: {best_ler:.4f}")
    print(f"Test LER (held out, greedy): {test_ler:.4f}")
    print("Test samples (label → pred [source]):")
    for label, pred, src in test_samples:
        match = "✓" if label == pred else " "
        print(f"  {match} {label!r:>20} → {pred!r:<25} [{src}]")

    summary = {
        "best_val_ler": best_ler,
        "test_ler_greedy": test_ler,
        "history": history,
        "test_samples_greedy": [
            {"label": l, "pred": p, "source": s} for l, p, s in test_samples
        ],
    }
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    latest = os.path.join("experiments", "latest_ctc_v2")
    if os.path.islink(latest) or os.path.exists(latest):
        if os.path.islink(latest):
            os.remove(latest)
    os.symlink(os.path.abspath(exp_dir), latest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=400,
                        help="With weighted sampling, defines an 'epoch' as N batches.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()
    train(args)
