"""
Dataset classes for ASL data loading.

Phase 1: ASLAlphabetDataset - loads landmark features for static fingerspelling.
Phase 2: WLASLDataset - loads temporal landmark sequences for word-level signs.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.landmark_extractor import normalize_landmarks


_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ASLAlphabetDataset(Dataset):
    """Dataset for ASL Alphabet static landmark features (Phase 1).

    Expects landmarks extracted by landmark_extractor.py at:
        landmarks_dir/<CLASS>.npy  — shape (N, 63) float32

    and a class map at:
        <parent of landmarks_dir>/class_map.json

    Only A–Z classes are loaded; non-letter classes (del, nothing, space) are
    excluded so the model never predicts them during inference.
    """

    def __init__(self, landmarks_dir, split="train", val_fraction=0.2, seed=42, transform=None):
        self.transform = transform

        class_map_path = os.path.join(os.path.dirname(landmarks_dir), "class_map.json")
        if not os.path.isfile(class_map_path):
            raise FileNotFoundError(
                f"class_map.json not found at '{class_map_path}'. "
                "Run 'make extract-landmarks' first."
            )
        with open(class_map_path) as f:
            raw_class_map = json.load(f)

        # Keep only A–Z, re-index contiguously so class indices are 0–25
        letter_names = sorted(k for k in raw_class_map if k.upper() in _ALPHABET)
        self.class_map = {name.upper(): idx for idx, name in enumerate(letter_names)}
        self.idx_to_class = {v: k for k, v in self.class_map.items()}

        all_features = []
        all_labels = []

        for class_name, class_idx in sorted(self.class_map.items(), key=lambda x: x[1]):
            orig_name = next(n for n in raw_class_map if n.upper() == class_name)
            npy_path = os.path.join(landmarks_dir, f"{orig_name}.npy")
            if not os.path.isfile(npy_path):
                continue
            features = np.load(npy_path)  # (N, 63)
            features = np.array([normalize_landmarks(row) for row in features], dtype=np.float32)
            labels = np.full(len(features), class_idx, dtype=np.int64)
            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            raise RuntimeError(
                f"No .npy files found in '{landmarks_dir}'. "
                "Run 'make extract-landmarks' first."
            )

        features_np = np.concatenate(all_features, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(features_np))
        split_at = int(len(indices) * (1.0 - val_fraction))

        if split == "train":
            idx = indices[:split_at]
        else:
            idx = indices[split_at:]

        self.features = torch.from_numpy(features_np[idx])
        self.labels = torch.from_numpy(labels_np[idx])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    @property
    def num_classes(self):
        return len(self.class_map)


class WLASLDataset(Dataset):
    """Dataset for WLASL video landmark sequences (Phase 2)."""

    def __init__(self, landmarks_dir, split="train", max_seq_len=64, transform=None):
        raise NotImplementedError("Phase 2 - implement WLASL dataset")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
