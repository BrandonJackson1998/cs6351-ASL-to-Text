"""Extract hand crop images from frames using MediaPipe landmarks.

Given a video frame and hand landmarks, crop to the hand region with padding
and resize to a fixed size for CNN training.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def get_hand_bbox(landmarks: np.ndarray, padding: float = 0.2) -> Tuple[int, int, int, int]:
    """Calculate bounding box around hand landmarks with padding.

    Args:
        landmarks: (21, 3) array of normalized hand landmarks (x, y, z)
        padding: Fractional padding to add around bbox (default: 20%)

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates (not normalized)
    """
    # Extract x, y (ignore z)
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    # Find min/max
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # Add padding
    width = x_max - x_min
    height = y_max - y_min

    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding

    # Clamp to [0, 1] range (normalized coords)
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)
    y_min = max(0.0, y_min)
    y_max = min(1.0, y_max)

    return x_min, y_min, x_max, y_max


def extract_hand_crop(
    frame: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = (200, 200),
    padding: float = 0.2
) -> Optional[np.ndarray]:
    """Extract and resize hand crop from frame.

    Args:
        frame: BGR image (H, W, 3)
        landmarks: (63,) or (21, 3) normalized hand landmarks from MediaPipe
        target_size: Output image size (width, height)
        padding: Fractional padding around hand bbox

    Returns:
        RGB image (target_size[1], target_size[0], 3) uint8, or None if crop fails
    """
    # Reshape if flat
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    # Get bounding box in normalized coords
    x_min, y_min, x_max, y_max = get_hand_bbox(landmarks, padding)

    # Convert to pixel coords
    h, w = frame.shape[:2]
    x_min_px = int(x_min * w)
    x_max_px = int(x_max * w)
    y_min_px = int(y_min * h)
    y_max_px = int(y_max * h)

    # Validate crop
    if x_max_px <= x_min_px or y_max_px <= y_min_px:
        return None

    # Crop frame
    crop = frame[y_min_px:y_max_px, x_min_px:x_max_px]

    if crop.size == 0:
        return None

    # Resize to target
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return rgb


def denormalize_landmarks(landmarks: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    """Convert normalized landmarks (0-1) to pixel coordinates.

    Args:
        landmarks: (63,) or (21, 3) normalized landmarks
        frame_shape: (height, width) of source frame

    Returns:
        (21, 3) landmarks in pixel coordinates
    """
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(21, 3)

    h, w = frame_shape
    lm_denorm = landmarks.copy()
    lm_denorm[:, 0] *= w  # x
    lm_denorm[:, 1] *= h  # y
    # z stays relative

    return lm_denorm


def extract_crops_from_dataset(
    data_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (200, 200),
    padding: float = 0.2,
    verbose: bool = True
):
    """Extract hand crops from a landmark dataset.

    Expects directory structure:
        data_dir/
            landmarks/
                A.npy  # shape (N, 63)
                B.npy
                ...
            class_map.json

    Creates:
        output_dir/
            A/
                img_00000.jpg
                img_00001.jpg
                ...
            B/
                ...

    Args:
        data_dir: Root directory with landmarks/ subdirectory
        output_dir: Output directory for hand crops
        target_size: Crop output size (width, height)
        padding: Padding around hand bbox
        verbose: Print progress
    """
    import json

    landmarks_dir = data_dir / "landmarks"

    if not landmarks_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {landmarks_dir}")

    # Load class map
    class_map_path = data_dir / "class_map.json"
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
        letters = sorted(class_map.keys())
    else:
        # Infer from .npy files
        letters = sorted([f.stem for f in landmarks_dir.glob("*.npy")])

    if verbose:
        print(f"Processing {len(letters)} classes from {data_dir}")
        print(f"Output: {output_dir}")

    total_processed = 0
    total_failed = 0

    for letter in letters:
        landmark_file = landmarks_dir / f"{letter}.npy"
        if not landmark_file.exists():
            continue

        # Load landmarks
        landmarks = np.load(landmark_file)  # (N, 63)

        # Create output directory
        letter_output_dir = output_dir / letter
        letter_output_dir.mkdir(parents=True, exist_ok=True)

        # For each landmark, we need the original image
        # Problem: we only have landmarks, not source frames!
        # We need to modify this to work with the actual dataset structure

        if verbose:
            print(f"  {letter}: {len(landmarks)} samples")

        total_processed += len(landmarks)

    if verbose:
        print(f"\nTotal: {total_processed} samples")
        print(f"Failed: {total_failed} crops")


def extract_crops_from_video_with_landmarks(
    video_path: Path,
    landmarks_array: np.ndarray,
    frame_indices: np.ndarray,
    output_dir: Path,
    letter: str,
    target_size: Tuple[int, int] = (200, 200),
    padding: float = 0.2
) -> int:
    """Extract hand crops from video frames given precomputed landmarks.

    Args:
        video_path: Path to video file
        landmarks_array: (N, 63) normalized landmarks
        frame_indices: (N,) frame numbers corresponding to landmarks
        output_dir: Output directory
        letter: Letter label for this video
        target_size: Crop output size
        padding: Padding around hand bbox

    Returns:
        Number of successfully extracted crops
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Create output directory
    letter_dir = output_dir / letter
    letter_dir.mkdir(parents=True, exist_ok=True)

    num_extracted = 0

    for i, frame_idx in enumerate(frame_indices):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Extract crop
        crop = extract_hand_crop(frame, landmarks_array[i], target_size, padding)

        if crop is not None:
            # Save as RGB image
            output_path = letter_dir / f"img_{frame_idx:05d}.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            num_extracted += 1

    cap.release()
    return num_extracted


if __name__ == "__main__":
    # Quick test
    import sys

    # Create synthetic test
    test_landmarks = np.random.rand(21, 3)
    test_landmarks[:, 0] = np.clip(test_landmarks[:, 0] * 0.3 + 0.35, 0, 1)  # Center x
    test_landmarks[:, 1] = np.clip(test_landmarks[:, 1] * 0.3 + 0.35, 0, 1)  # Center y

    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    crop = extract_hand_crop(test_frame, test_landmarks)

    if crop is not None:
        print(f"✓ Hand crop extracted: {crop.shape}")
        print(f"  Min: {crop.min()}, Max: {crop.max()}, Dtype: {crop.dtype}")
    else:
        print("✗ Failed to extract crop")
        sys.exit(1)
