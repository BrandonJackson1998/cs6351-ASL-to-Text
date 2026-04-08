"""
Dataset classes for ASL data loading.

Phase 1: ASLAlphabetDataset - loads landmark features for static fingerspelling.
Phase 2: WLASLDataset - loads temporal landmark sequences for word-level signs.
"""

from torch.utils.data import Dataset


class ASLAlphabetDataset(Dataset):
    """Dataset for ASL Alphabet static landmark features (Phase 1)."""

    def __init__(self, landmarks_dir, split="train", transform=None):
        raise NotImplementedError("Phase 1 - implement alphabet dataset")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class WLASLDataset(Dataset):
    """Dataset for WLASL video landmark sequences (Phase 2)."""

    def __init__(self, landmarks_dir, split="train", max_seq_len=64, transform=None):
        raise NotImplementedError("Phase 2 - implement WLASL dataset")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
