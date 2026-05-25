"""Landmark-level augmentation for ASL fingerspelling training.

All augmentations operate on (63,) float32 vectors representing 21 hand
keypoints x (x, y, z), already wrist-centered and scale-normalized by
landmark_extractor.normalize_landmarks.
"""

import numpy as np


def jitter(landmarks, sigma=0.02, rng=None):
    rng = rng or np.random
    return landmarks + rng.normal(0.0, sigma, size=landmarks.shape).astype(np.float32)


def mirror_x(landmarks):
    pts = landmarks.reshape(21, 3).copy()
    pts[:, 0] *= -1
    return pts.flatten().astype(np.float32)


def rotate_3d(landmarks, max_deg=15.0, rng=None):
    rng = rng or np.random
    pts = landmarks.reshape(21, 3).astype(np.float64)
    angles = rng.uniform(-max_deg, max_deg, size=3) * np.pi / 180.0
    cx, sx = np.cos(angles[0]), np.sin(angles[0])
    cy, sy = np.cos(angles[1]), np.sin(angles[1])
    cz, sz = np.cos(angles[2]), np.sin(angles[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rotated = pts @ R.T
    return rotated.flatten().astype(np.float32)


def random_augment(landmarks, p_jitter=0.7, p_rotate=0.5, p_mirror=0.5, rng=None):
    rng = rng or np.random
    out = landmarks
    if rng.random() < p_jitter:
        out = jitter(out, sigma=0.02, rng=rng)
    if rng.random() < p_rotate:
        out = rotate_3d(out, max_deg=15.0, rng=rng)
    if rng.random() < p_mirror:
        out = mirror_x(out)
    return out
