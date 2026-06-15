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
from src.preprocessing.augment import random_augment


_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _stratified_split_indices(labels, fractions=(0.6, 0.2, 0.2), seed=42):
    """Return (train_idx, val_idx, test_idx) stratified by class."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * fractions[0])
        n_val = int(n * fractions[1])
        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


class ASLAlphabetDataset(Dataset):
    """Dataset for ASL Alphabet static landmark features (Phase 1).

    Expects landmarks extracted by landmark_extractor.py at:
        landmarks_dir/<CLASS>.npy  — shape (N, 63) float32

    and a class map at:
        <parent of landmarks_dir>/class_map.json

    Only A–Z classes are loaded; non-letter classes (del, nothing, space) are
    excluded so the model never predicts them during inference.

    Stratified 60/20/20 train/val/test split, fixed seed.
    """

    def __init__(
        self,
        landmarks_dir,
        split="train",
        fractions=(0.6, 0.2, 0.2),
        seed=42,
        augment=False,
        transform=None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        self.split = split
        self.augment = augment and split == "train"
        self.transform = transform
        self._aug_rng = np.random.default_rng(seed + hash(split) % 1000)

        class_map_path = os.path.join(os.path.dirname(landmarks_dir), "class_map.json")
        if not os.path.isfile(class_map_path):
            raise FileNotFoundError(
                f"class_map.json not found at '{class_map_path}'. "
                "Run 'make extract-landmarks' first."
            )
        with open(class_map_path) as f:
            raw_class_map = json.load(f)

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
            features = np.load(npy_path)
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

        train_idx, val_idx, test_idx = _stratified_split_indices(
            labels_np, fractions=fractions, seed=seed
        )
        idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        idx = idx_map[split]

        self.features_np = features_np[idx]
        self.labels_np = labels_np[idx]
        self.features = torch.from_numpy(self.features_np)
        self.labels = torch.from_numpy(self.labels_np)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.augment:
            x_np = random_augment(self.features_np[idx], rng=self._aug_rng)
            x = torch.from_numpy(x_np)
        else:
            x = self.features[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def get_arrays(self, augment=False):
        """Return (X, y) numpy arrays for sklearn-style consumers.

        If augment=True (and split=='train'), each sample is replaced with one
        augmented copy. For PPCA training we typically call this on train with
        augmentation, and on val/test without.
        """
        if augment and self.split == "train":
            X = np.stack([random_augment(x, rng=self._aug_rng) for x in self.features_np])
        else:
            X = self.features_np.copy()
        return X, self.labels_np.copy()

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
