"""Evaluate CTC v2 + PPCA shallow fusion.

For each sequence:
  1. Run CTC v2 → (T, 27) log-probs
  2. Run PPCA-Mixture per-frame  → (T, 26) log-likelihoods
  3. Fuse with weight lam
  4. Decode (greedy or beam+LM)

Sweeps lam on dev only, reports test once at the chosen lam.
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
from src.models.decode_ctc import (
    greedy_decode, beam_search_decode_lm, fuse_with_ppca,
)
from src.models.char_lm import build_default_lm
from src.models.ppca_classifier import PPCAMixtureClassifier
from src.preprocessing.combined_dataset import CombinedFSDataset, NUM_CLASSES, collate


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


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


def compute_ppca_emissions_for_seq(seq, mixture, class_map):
    """Return a (T, 26) matrix of PPCA log-likelihoods, one column per letter A..Z."""
    Xs = mixture.scaler.transform(seq.astype(np.float64))
    T = Xs.shape[0]
    out = np.full((T, 26), -np.inf, dtype=np.float64)
    for cls in mixture.classes_:
        ll = mixture.ppcas_[int(cls)].log_likelihood(Xs)
        # Map dataset-class-index → letter A-Z via class_map
        letter = None
        for name, idx in class_map.items():
            if int(idx) == int(cls):
                letter = name.upper()
                break
        if letter and letter in _LETTERS:
            out[:, _LETTERS.index(letter)] = ll
    if np.any(np.isinf(out)):
        floor = out[np.isfinite(out)].min()
        out[np.isinf(out)] = floor
    return out


def collect_split(model, mixture, class_map, loader, device):
    """Run CTC + PPCA on every sequence in a split. Returns list of dicts."""
    items = []
    model.eval()
    with torch.no_grad():
        for seqs, seq_lens, _, _, labels, sources in loader:
            seqs_dev = seqs.to(device)
            log_probs = model(seqs_dev, seq_lens).cpu().numpy()  # (T, B, V)
            for i, (label, src) in enumerate(zip(labels, sources)):
                t = int(seq_lens[i])
                seq_np = seqs[i, :t].cpu().numpy()
                ctc_lp = log_probs[:t, i, :]
                ppca_ll = compute_ppca_emissions_for_seq(seq_np, mixture, class_map)
                items.append({
                    "label": label, "source": src,
                    "ctc_lp": ctc_lp, "ppca_ll": ppca_ll,
                })
    return items


def ler_with(items, decoder_fn):
    total_chars, total_errs = 0, 0
    for it in items:
        pred = decoder_fn(it)
        total_errs += levenshtein(pred, it["label"])
        total_chars += max(1, len(it["label"]))
    return total_errs / max(1, total_chars)


def make_greedy_decoder(lam: float):
    def dec(it):
        fused = fuse_with_ppca(it["ctc_lp"], it["ppca_ll"], lam)
        return greedy_decode(fused)
    return dec


def make_beamlm_decoder(lam: float, char_lm, alpha: float, beta: float, beam_size: int):
    def dec(it):
        fused = fuse_with_ppca(it["ctc_lp"], it["ppca_ll"], lam)
        hyps = beam_search_decode_lm(fused, char_lm, alpha=alpha, beta=beta, beam_size=beam_size)
        return hyps[0] if hyps else ""
    return dec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctc-checkpoint", default="experiments/latest_ctc_v2/best_model.pt")
    parser.add_argument("--mixture-dir", default="experiments/latest/mixture")
    parser.add_argument("--class-map", default="experiments/latest/class_map.json")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0])
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

    print(f"Loading PPCA-Mixture from {args.mixture_dir}")
    mixture = PPCAMixtureClassifier.load(args.mixture_dir)
    with open(args.class_map) as f:
        class_map = json.load(f)

    val_loader = DataLoader(CombinedFSDataset("val"), batch_size=32, shuffle=False,
                            num_workers=0, collate_fn=collate)
    test_loader = DataLoader(CombinedFSDataset("test"), batch_size=32, shuffle=False,
                             num_workers=0, collate_fn=collate)

    print("Computing CTC + PPCA per-frame outputs on dev...")
    dev_items = collect_split(model, mixture, class_map, val_loader, device)
    print("Computing CTC + PPCA per-frame outputs on test...")
    test_items = collect_split(model, mixture, class_map, test_loader, device)

    print("\nBuilding character LM...")
    char_lm = build_default_lm(n=args.lm_n)

    # ------------------------------------------------------------------
    # Sweep lambda on dev (greedy first, fast)
    # ------------------------------------------------------------------
    print("\n=== DEV sweep (greedy decoder) ===")
    best_greedy = {"ler": float("inf"), "lam": None}
    for lam in args.lambdas:
        ler = ler_with(dev_items, make_greedy_decoder(lam))
        flag = "  <" if ler < best_greedy["ler"] else ""
        print(f"  lam={lam:.2f}  dev_LER={ler:.4f}{flag}")
        if ler < best_greedy["ler"]:
            best_greedy = {"ler": ler, "lam": lam}

    print(f"\nBest dev (greedy): lam={best_greedy['lam']}  LER={best_greedy['ler']:.4f}")

    # ------------------------------------------------------------------
    # Lambda sweep with beam+LM (slower but matters most)
    # ------------------------------------------------------------------
    print("\n=== DEV sweep (beam + LM, alpha=%.2f beta=%.2f) ===" % (args.lm_alpha, args.lm_beta))
    best_beam = {"ler": float("inf"), "lam": None}
    for lam in args.lambdas:
        dec = make_beamlm_decoder(lam, char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
        ler = ler_with(dev_items, dec)
        flag = "  <" if ler < best_beam["ler"] else ""
        print(f"  lam={lam:.2f}  dev_LER={ler:.4f}{flag}")
        if ler < best_beam["ler"]:
            best_beam = {"ler": ler, "lam": lam}

    print(f"\nBest dev (beam+LM): lam={best_beam['lam']}  LER={best_beam['ler']:.4f}")

    # ------------------------------------------------------------------
    # TEST eval — single shot at the dev-chosen lambdas
    # ------------------------------------------------------------------
    print("\n=== TEST EVAL (single shot at dev-chosen lambdas) ===")

    test_greedy_baseline = ler_with(test_items, make_greedy_decoder(0.0))
    test_greedy_fused = ler_with(test_items, make_greedy_decoder(best_greedy["lam"]))
    test_beamlm_baseline = ler_with(
        test_items, make_beamlm_decoder(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    )
    test_beamlm_fused = ler_with(
        test_items, make_beamlm_decoder(best_beam["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    )

    print(f"Test LER (greedy, lam=0):              {test_greedy_baseline:.4f}")
    print(f"Test LER (greedy, lam={best_greedy['lam']}):    {test_greedy_fused:.4f}")
    print(f"Test LER (beam+LM, lam=0):             {test_beamlm_baseline:.4f}")
    print(f"Test LER (beam+LM, lam={best_beam['lam']}):  {test_beamlm_fused:.4f}")

    # Per-source breakdown at the fused beam+LM setting
    print("\nPer-source test LER (beam+LM):")
    for src in ("fswild", "kaggle"):
        sub = [it for it in test_items if it["source"] == src]
        if not sub:
            continue
        b = ler_with(sub, make_beamlm_decoder(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
        f_ = ler_with(sub, make_beamlm_decoder(best_beam["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size))
        print(f"  {src:8s}  n={len(sub):4d}  baseline={b:.4f}  fused(lam={best_beam['lam']})={f_:.4f}")

    print("\nSample test predictions (label → baseline / fused [source]):")
    base_dec = make_beamlm_decoder(0.0, char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    fused_dec = make_beamlm_decoder(best_beam["lam"], char_lm, args.lm_alpha, args.lm_beta, args.beam_size)
    for i, it in enumerate(test_items[:25]):
        b = base_dec(it); f_ = fused_dec(it)
        m1 = "✓" if b == it["label"] else " "
        m2 = "✓" if f_ == it["label"] else " "
        print(f"  {it['label']!r:>20} → {m1} {b!r:<22} | {m2} {f_!r:<22} [{it['source']}]")

    out = {
        "best_dev_greedy_lam": best_greedy["lam"],
        "best_dev_greedy_ler": best_greedy["ler"],
        "best_dev_beam_lam": best_beam["lam"],
        "best_dev_beam_ler": best_beam["ler"],
        "test_ler_greedy_baseline": test_greedy_baseline,
        "test_ler_greedy_fused": test_greedy_fused,
        "test_ler_beam_baseline": test_beamlm_baseline,
        "test_ler_beam_fused": test_beamlm_fused,
    }
    save_path = os.path.join(os.path.dirname(args.ctc_checkpoint), "fusion_results.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
