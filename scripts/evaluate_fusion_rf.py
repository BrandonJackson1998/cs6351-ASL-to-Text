"""Shallow fusion of CTC v2 with a Random Forest per-frame letter classifier.

Same setup as evaluate_fusion.py but the emission model is an sklearn
RandomForestClassifier saved by train_rf.py. RF gives us calibrated class
probabilities directly via predict_proba.

Sweeps lambda on dev only; reports test once.
"""

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.ctc_lstm_v2 import CTCLSTMv2
from src.models.decode_ctc import (
    greedy_decode, beam_search_decode_lm, fuse_with_ppca,
)
from src.models.char_lm import build_default_lm
from src.preprocessing.combined_dataset import CombinedFSDataset, NUM_CLASSES, collate


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EPS = 1e-8


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


def rf_log_emissions(rf, class_map, seq):
    """Return (T, 26) log-probabilities for letters A..Z under the RF.

    The RF was trained over the dataset's class indices; we remap to A..Z order.
    """
    proba = rf.predict_proba(seq.astype(np.float32))  # (T, num_dataset_classes)
    rf_classes = list(rf.classes_)
    out = np.full((seq.shape[0], 26), -10.0, dtype=np.float64)  # log-floor
    for class_name, class_idx in class_map.items():
        letter = class_name.upper()
        if letter not in _LETTERS:
            continue
        if class_idx not in rf_classes:
            continue
        col = _LETTERS.index(letter)
        rf_col = rf_classes.index(class_idx)
        out[:, col] = np.log(proba[:, rf_col] + _EPS)
    return out


def collect_split(model, rf, class_map, loader, device):
    items = []
    model.eval()
    with torch.no_grad():
        for seqs, seq_lens, _, _, labels, sources in loader:
            seqs_dev = seqs.to(device)
            log_probs = model(seqs_dev, seq_lens).cpu().numpy()
            for i, (label, src) in enumerate(zip(labels, sources)):
                t = int(seq_lens[i])
                seq_np = seqs[i, :t].cpu().numpy()
                ctc_lp = log_probs[:t, i, :]
                rf_lp = rf_log_emissions(rf, class_map, seq_np)
                items.append({"label": label, "source": src,
                              "ctc_lp": ctc_lp, "ppca_ll": rf_lp})
    return items


def ler_with(items, decoder_fn):
    total_chars, total_errs = 0, 0
    for it in items:
        pred = decoder_fn(it)
        total_errs += levenshtein(pred, it["label"])
        total_chars += max(1, len(it["label"]))
    return total_errs / max(1, total_chars)


def make_greedy(lam):
    def dec(it):
        fused = fuse_with_ppca(it["ctc_lp"], it["ppca_ll"], lam)
        return greedy_decode(fused)
    return dec


def make_beam_lm(lam, char_lm, alpha, beta, beam_size):
    def dec(it):
        fused = fuse_with_ppca(it["ctc_lp"], it["ppca_ll"], lam)
        hyps = beam_search_decode_lm(fused, char_lm, alpha=alpha, beta=beta, beam_size=beam_size)
        return hyps[0] if hyps else ""
    return dec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--rf-dir", default="experiments/rf_fswild",
                        help="Dir containing model.pkl + class_map.json")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--lm-alpha", type=float, default=0.25)
    parser.add_argument("--lm-beta", type=float, default=0.0)
    parser.add_argument("--lm-n", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading CTC v2 from {args.ctc_checkpoint}")
    ckpt = torch.load(args.ctc_checkpoint, map_location=device, weights_only=False)
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

    print(f"Loading RF from {args.rf_dir}")
    with open(os.path.join(args.rf_dir, "model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(args.rf_dir, "class_map.json")) as f:
        class_map = json.load(f)

    val_loader = DataLoader(CombinedFSDataset("val"), batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=collate)
    test_loader = DataLoader(CombinedFSDataset("test"), batch_size=32, shuffle=False,
                             num_workers=0, collate_fn=collate)

    print("Computing CTC + RF per-frame outputs on dev...")
    dev_items = collect_split(model, rf, class_map, val_loader, device)
    print("Computing CTC + RF per-frame outputs on test...")
    test_items = collect_split(model, rf, class_map, test_loader, device)

    print("\nBuilding character LM...")
    char_lm = build_default_lm(n=args.lm_n)

    print("\n=== DEV sweep (greedy decoder) ===")
    best_g = {"ler": float("inf"), "lam": None}
    for lam in args.lambdas:
        ler = ler_with(dev_items, make_greedy(lam))
        flag = "  <" if ler < best_g["ler"] else ""
        print(f"  lam={lam:.2f}  dev_LER={ler:.4f}{flag}")
        if ler < best_g["ler"]:
            best_g = {"ler": ler, "lam": lam}
    print(f"\nBest dev (greedy): lam={best_g['lam']}  LER={best_g['ler']:.4f}")

    print("\n=== DEV sweep (beam + LM) ===")
    best_b = {"ler": float("inf"), "lam": None}
    for lam in args.lambdas:
        dec = make_beam_lm(lam, char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
        ler = ler_with(dev_items, dec)
        flag = "  <" if ler < best_b["ler"] else ""
        print(f"  lam={lam:.2f}  dev_LER={ler:.4f}{flag}")
        if ler < best_b["ler"]:
            best_b = {"ler": ler, "lam": lam}
    print(f"\nBest dev (beam+LM): lam={best_b['lam']}  LER={best_b['ler']:.4f}")

    print("\n=== TEST EVAL (single shot) ===")
    test_g_base = ler_with(test_items, make_greedy(0.0))
    test_g_fuse = ler_with(test_items, make_greedy(best_g["lam"]))
    test_b_base = ler_with(test_items, make_beam_lm(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
    test_b_fuse = ler_with(test_items, make_beam_lm(best_b["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
    print(f"Test LER (greedy, lam=0):              {test_g_base:.4f}")
    print(f"Test LER (greedy, lam={best_g['lam']}):    {test_g_fuse:.4f}")
    print(f"Test LER (beam+LM, lam=0):             {test_b_base:.4f}")
    print(f"Test LER (beam+LM, lam={best_b['lam']}):  {test_b_fuse:.4f}")

    print("\nPer-source test LER (beam+LM):")
    for src in ("fswild", "kaggle"):
        sub = [it for it in test_items if it["source"] == src]
        if not sub:
            continue
        b = ler_with(sub, make_beam_lm(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
        f_ = ler_with(sub, make_beam_lm(best_b["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
        print(f"  {src:8s}  n={len(sub):4d}  baseline={b:.4f}  fused(lam={best_b['lam']})={f_:.4f}")

    print("\nSample test predictions (label → baseline / fused [source]):")
    base_dec = make_beam_lm(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    fused_dec = make_beam_lm(best_b["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    for it in test_items[:25]:
        b = base_dec(it); f_ = fused_dec(it)
        m1 = "✓" if b == it["label"] else " "
        m2 = "✓" if f_ == it["label"] else " "
        print(f"  {it['label']!r:>20} → {m1} {b!r:<22} | {m2} {f_!r:<22} [{it['source']}]")

    out = {
        "best_dev_greedy_lam": best_g["lam"],
        "best_dev_greedy_ler": best_g["ler"],
        "best_dev_beam_lam": best_b["lam"],
        "best_dev_beam_ler": best_b["ler"],
        "test_ler_greedy_baseline": test_g_base,
        "test_ler_greedy_fused": test_g_fuse,
        "test_ler_beam_baseline": test_b_base,
        "test_ler_beam_fused": test_b_fuse,
    }
    save_path = os.path.join(args.rf_dir, "fusion_results.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
