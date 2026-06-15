"""Combined fingerspelling sequence dataset.

Merges:
  - ChicagoFSWild train/dev/test (real video sequences with letter labels)
  - Kaggle ASL Alphabet (single-frame letter exemplars, treated as length-1
    fingerspelled sequences)

The Kaggle data is split 80/10/10 with a fixed seed and the Kaggle splits are
concatenated into the corresponding FSWild splits. Test is touched only at
final evaluation.
"""

import json
import os
import string
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.landmark_extractor import normalize_landmarks


_LETTERS = list(string.ascii_uppercase)
BLANK_IDX = 0  # CTC convention: 0 = blank
LETTER_TO_TARGET = {c: i + 1 for i, c in enumerate(_LETTERS)}
NUM_CLASSES = 1 + len(_LETTERS)  # 27


def label_to_targets(label: str) -> torch.Tensor:
    return torch.tensor(
        [LETTER_TO_TARGET[c] for c in label.upper() if c in LETTER_TO_TARGET],
        dtype=torch.long,
    )


def targets_to_label(targets) -> str:
    return "".join(
        _LETTERS[t - 1] for t in targets if t != BLANK_IDX and 1 <= t <= len(_LETTERS)
    )


def _kaggle_split_indices(n_total: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val = int(n_total * 0.1)
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return train, val, test


def load_kaggle_sequences(landmarks_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load Kaggle alphabet landmarks as length-1 sequences."""
    class_map_path = os.path.join(os.path.dirname(landmarks_dir), "class_map.json")
    with open(class_map_path) as f:
        raw = json.load(f)

    sequences: List[np.ndarray] = []
    labels: List[str] = []
    for name, _idx in raw.items():
        letter = name.upper()
        if letter not in _LETTERS:
            continue
        npy_path = os.path.join(landmarks_dir, f"{name}.npy")
        if not os.path.isfile(npy_path):
            continue
        feats = np.load(npy_path).astype(np.float32)
        feats = np.array([normalize_landmarks(row) for row in feats], dtype=np.float32)
        for row in feats:
            sequences.append(row[None, :].astype(np.float32))  # (1, 63)
            labels.append(letter)
    return sequences, labels


class CombinedFSDataset(Dataset):
    """Fingerspelling dataset combining FSWild + Kaggle.

    Args:
        split: 'train' | 'val' | 'test'
        fswild_dir: directory containing train.npz / dev.npz / test.npz
        kaggle_dir: directory containing landmarks/ subdir + class_map.json
        augment: callable(np.ndarray) -> np.ndarray applied to each sequence
                 at __getitem__ time. Should be a no-op for val/test.
    """

    SPLIT_TO_FSWILD_NAME = {"train": "train", "val": "dev", "test": "test"}

    def __init__(
        self,
        split: str,
        fswild_dir: str = "data/chicagofswild/landmarks",
        kaggle_dir: str = "data/asl_alphabet",
        augment=None,
        kaggle_seed: int = 42,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.augment = augment if split == "train" else None

        self.sequences: List[np.ndarray] = []
        self.labels: List[str] = []
        self.sources: List[str] = []  # "fswild" or "kaggle"

        # FSWild
        fswild_name = self.SPLIT_TO_FSWILD_NAME[split]
        fswild_path = os.path.join(fswild_dir, f"{fswild_name}.npz")
        if os.path.isfile(fswild_path):
            data = np.load(fswild_path, allow_pickle=True)
            for s, l in zip(data["sequences"], data["labels"]):
                seq = s.astype(np.float32)
                if len(seq) < 1:
                    continue
                self.sequences.append(seq)
                self.labels.append(str(l))
                self.sources.append("fswild")

        # Kaggle (split into train/val/test by index)
        kag_landmarks_dir = os.path.join(kaggle_dir, "landmarks")
        if os.path.isdir(kag_landmarks_dir):
            kag_seqs, kag_labels = load_kaggle_sequences(kag_landmarks_dir)
            if kag_seqs:
                train_idx, val_idx, test_idx = _kaggle_split_indices(len(kag_seqs), seed=kaggle_seed)
                pick = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
                for i in pick:
                    self.sequences.append(kag_seqs[i])
                    self.labels.append(kag_labels[i])
                    self.sources.append("kaggle")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if self.augment is not None:
            seq = self.augment(seq)
        seq_t = torch.from_numpy(np.asarray(seq, dtype=np.float32))
        targets = label_to_targets(self.labels[idx])
        return seq_t, targets, self.labels[idx], self.sources[idx]

    def stats(self) -> dict:
        from collections import Counter
        src = Counter(self.sources)
        seq_lens = [len(s) for s in self.sequences]
        lab_lens = [len(l) for l in self.labels]
        return {
            "n": len(self.labels),
            "by_source": dict(src),
            "seq_len_min": min(seq_lens),
            "seq_len_mean": float(np.mean(seq_lens)),
            "seq_len_max": max(seq_lens),
            "label_len_min": min(lab_lens),
            "label_len_mean": float(np.mean(lab_lens)),
            "label_len_max": max(lab_lens),
        }


def collate(batch):
    """Pad-collate for Bi-LSTM + CTC training."""
    seqs, targets, labels, sources = zip(*batch)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    seqs_padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    targets_concat = torch.cat(targets)
    return seqs_padded, seq_lengths, targets_concat, target_lengths, list(labels), list(sources)
