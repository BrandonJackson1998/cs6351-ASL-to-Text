"""Temporal + landmark augmentation for fingerspelling sequences.

All operate on (T, 63) float32 sequences of normalized landmarks.

Augmentations:
  - time_warp: resample at 0.7-1.3x to simulate signing speed variation
  - frame_dropout: randomly drop a fraction of frames
  - jitter: per-coord Gaussian noise
  - rotate_3d: random 3D rotation around wrist
  - mirror_x: flip left-right
"""

import numpy as np


def time_warp(seq: np.ndarray, rate: float) -> np.ndarray:
    """Resample a sequence at the given rate (rate>1 = compress, rate<1 = stretch)."""
    T = len(seq)
    if T <= 1 or rate <= 0:
        return seq
    new_T = max(1, int(round(T / rate)))
    if new_T == T:
        return seq
    src_idx = np.linspace(0, T - 1, new_T)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (src_idx - lo).astype(np.float32)[:, None]
    return ((1 - frac) * seq[lo] + frac * seq[hi]).astype(np.float32)


def frame_dropout(seq: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Drop each frame with probability p; keep at least one frame."""
    if len(seq) <= 1:
        return seq
    keep = rng.random(len(seq)) > p
    if not keep.any():
        keep[0] = True
    return seq[keep]


def jitter(seq: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return seq + rng.normal(0.0, sigma, size=seq.shape).astype(np.float32)


def mirror_x(seq: np.ndarray) -> np.ndarray:
    pts = seq.reshape(-1, 21, 3).copy()
    pts[..., 0] *= -1
    return pts.reshape(-1, 63).astype(np.float32)


def rotate_3d(seq: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    pts = seq.reshape(-1, 21, 3).astype(np.float64)
    angles = rng.uniform(-max_deg, max_deg, size=3) * np.pi / 180.0
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rotated = pts @ R.T
    return rotated.reshape(-1, 63).astype(np.float32)


class FingerspellingAugment:
    """Stateful augmentation pipeline. Use one instance per worker for reproducibility."""

    def __init__(
        self,
        seed: int = 0,
        p_time_warp: float = 0.6,
        warp_range=(0.7, 1.3),
        p_frame_dropout: float = 0.4,
        frame_dropout_rate: float = 0.05,
        p_jitter: float = 0.6,
        jitter_sigma: float = 0.015,
        p_rotate: float = 0.4,
        rotate_max_deg: float = 15.0,
        p_mirror: float = 0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.p_time_warp = p_time_warp
        self.warp_range = warp_range
        self.p_frame_dropout = p_frame_dropout
        self.frame_dropout_rate = frame_dropout_rate
        self.p_jitter = p_jitter
        self.jitter_sigma = jitter_sigma
        self.p_rotate = p_rotate
        self.rotate_max_deg = rotate_max_deg
        self.p_mirror = p_mirror

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        out = seq
        if self.rng.random() < self.p_time_warp:
            rate = float(self.rng.uniform(*self.warp_range))
            out = time_warp(out, rate)
        if self.rng.random() < self.p_frame_dropout:
            out = frame_dropout(out, self.frame_dropout_rate, self.rng)
        if self.rng.random() < self.p_rotate:
            out = rotate_3d(out, self.rotate_max_deg, self.rng)
        if self.rng.random() < self.p_jitter:
            out = jitter(out, self.jitter_sigma, self.rng)
        if self.rng.random() < self.p_mirror:
            out = mirror_x(out)
        return out
