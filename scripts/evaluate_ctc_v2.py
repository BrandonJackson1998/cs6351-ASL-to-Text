"""One-shot test evaluation of the v2 CTC model.

Reports test-set LER (FSWild test split + Kaggle test slice) under three
decoders:
  - greedy (argmax per frame)
  - beam search (no LM)
  - beam search + character n-gram LM

The LM hyperparameters (alpha, beta) are tuned on dev only.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.models.decode_ctc import greedy_decode, beam_search_decode, beam_search_decode_lm
from src.models.char_lm import build_default_lm
from src.preprocessing.combined_dataset import (
    CombinedFSDataset,
    NUM_CLASSES,
    collate,
)


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


def split_log_probs(model, loader, device):
    """Return list of (T_i, V) np arrays + label/source tuples."""
    out = []
    model.eval()
    with torch.no_grad():
        for seqs, seq_lens, _, _, labels, sources in loader:
            seqs = seqs.to(device)
            log_probs = model(seqs, seq_lens).cpu().numpy()  # (T, B, V)
            for i, (label, src) in enumerate(zip(labels, sources)):
                t = int(seq_lens[i])
                out.append((log_probs[:t, i, :], label, src))
    return out


def ler_with_decoder(items, decoder_fn):
    total_chars, total_errs = 0, 0
    for log_probs, label, _src in items:
        pred = decoder_fn(log_probs)
        total_errs += levenshtein(pred, label)
        total_chars += max(1, len(label))
    return total_errs / max(1, total_chars)


def tune_lm_on_dev(dev_items, char_lm, beam_size, alphas, betas):
    best = {"ler": float("inf"), "alpha": None, "beta": None}
    for alpha in alphas:
        for beta in betas:
            def dec(lp, alpha=alpha, beta=beta):
                hyps = beam_search_decode_lm(lp, char_lm, alpha=alpha, beta=beta,
                                             beam_size=beam_size)
                return hyps[0] if hyps else ""
            ler = ler_with_decoder(dev_items, dec)
            print(f"  dev   alpha={alpha:.2f} beta={beta:.2f}  LER={ler:.4f}")
            if ler < best["ler"]:
                best = {"ler": ler, "alpha": alpha, "beta": beta}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--lm-n", type=int, default=6)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 1.5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    model = CTCLSTMv2(
        input_dim=63,
        hidden_dim=train_args.get("hidden_dim", 256),
        num_layers=train_args.get("num_layers", 3),
        dropout=train_args.get("dropout", 0.3),
        num_classes=NUM_CLASSES,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_loader = DataLoader(
        CombinedFSDataset("val"),
        batch_size=32, shuffle=False, num_workers=0, collate_fn=collate,
    )
    test_loader = DataLoader(
        CombinedFSDataset("test"),
        batch_size=32, shuffle=False, num_workers=0, collate_fn=collate,
    )

    print("Computing log-probs on dev (for LM tuning)...")
    dev_items = split_log_probs(model, val_loader, device)
    print("Computing log-probs on test...")
    test_items = split_log_probs(model, test_loader, device)

    print("\nBuilding character LM...")
    char_lm = build_default_lm(n=args.lm_n)

    print("\nTuning LM weights on dev:")
    best = tune_lm_on_dev(dev_items, char_lm, args.beam_size, args.alphas, args.betas)
    print(f"\nBest dev: alpha={best['alpha']}  beta={best['beta']}  LER={best['ler']:.4f}")

    print("\n=== TEST EVAL (single shot, no further tuning) ===")

    # Greedy
    greedy_ler = ler_with_decoder(test_items, greedy_decode)
    print(f"Test LER (greedy):           {greedy_ler:.4f}")

    # Plain beam (no LM)
    def beam_top1(lp):
        hyps = beam_search_decode(lp, beam_size=args.beam_size)
        return hyps[0] if hyps else ""
    beam_ler = ler_with_decoder(test_items, beam_top1)
    print(f"Test LER (beam, no LM):      {beam_ler:.4f}")

    # Beam + LM (tuned weights)
    def beam_lm_top1(lp):
        hyps = beam_search_decode_lm(
            lp, char_lm, alpha=best["alpha"], beta=best["beta"], beam_size=args.beam_size
        )
        return hyps[0] if hyps else ""
    beam_lm_ler = ler_with_decoder(test_items, beam_lm_top1)
    print(f"Test LER (beam + char LM):   {beam_lm_ler:.4f}")

    # Per-source LER
    by_source = {}
    for items_name, items in [("test", test_items)]:
        for src in ("fswild", "kaggle"):
            sub = [it for it in items if it[2] == src]
            if not sub:
                continue
            sub_greedy = ler_with_decoder(sub, greedy_decode)
            sub_beam_lm = ler_with_decoder(sub, beam_lm_top1)
            by_source[src] = {"n": len(sub), "greedy": sub_greedy, "beam_lm": sub_beam_lm}
    print("\nPer-source test LER:")
    for src, d in by_source.items():
        print(f"  {src:8s}  n={d['n']}  greedy={d['greedy']:.4f}  beam+LM={d['beam_lm']:.4f}")

    # Sample qualitative predictions
    print("\nSample test predictions (label → greedy / beam+LM [source]):")
    for i, (lp, label, src) in enumerate(test_items[:25]):
        g = greedy_decode(lp)
        bl = beam_lm_top1(lp)
        m1 = "✓" if g == label else " "
        m2 = "✓" if bl == label else " "
        print(f"  {label!r:>20} → {m1} {g!r:<25} | {m2} {bl!r:<25} [{src}]")

    # Save summary
    out = {
        "checkpoint": args.checkpoint,
        "best_dev_lm_alpha": best["alpha"],
        "best_dev_lm_beta": best["beta"],
        "best_dev_ler": best["ler"],
        "test_ler_greedy": greedy_ler,
        "test_ler_beam_no_lm": beam_ler,
        "test_ler_beam_lm": beam_lm_ler,
        "per_source": by_source,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
