"""
Landmark Extractor
Extracts hand/pose landmarks from images and video using MediaPipe.

Phase 1: MediaPipe Tasks HandLandmarker (21 keypoints x 3 coords = 63 features).
Phase 2: MediaPipe Holistic (hands + pose + face) from video frames.
"""

import argparse
import json
import os
import sys

import contextlib
import cv2
import numpy as np

# Set MEDIAPIPE_VERBOSE=1 to see MediaPipe's C-level logs (suppressed by default).
_VERBOSE = os.environ.get("MEDIAPIPE_VERBOSE", "0") == "1"

@contextlib.contextmanager
def _silence_stderr():
    """Redirect C-level stderr to /dev/null unless MEDIAPIPE_VERBOSE=1."""
    if _VERBOSE:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

with _silence_stderr():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from tqdm import tqdm

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hand_landmarker.task")


def _download_model():
    """Download MediaPipe hand_landmarker model if not present."""
    import subprocess

    model_path = os.path.abspath(MODEL_PATH)
    model_dir = os.path.dirname(model_path)

    print(f"MediaPipe model not found at '{model_path}'")
    print("Downloading hand_landmarker.task (7.5MB) from Google MediaPipe...")

    os.makedirs(model_dir, exist_ok=True)

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    # Try curl first (more reliable on macOS)
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", model_path, url],
            capture_output=True,
            check=True
        )
        if os.path.isfile(model_path):
            print(f"✓ Downloaded to {model_path}")
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback to urllib
    try:
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Downloaded to {model_path}")
    except Exception as e:
        print(
            f"ERROR: Failed to download model: {e}\n"
            "Please download manually:\n"
            f"  curl -L -o {model_path} {url}",
            file=sys.stderr,
        )
        sys.exit(1)


def _make_landmarker():
    model_path = os.path.abspath(MODEL_PATH)
    if not os.path.isfile(model_path):
        _download_model()
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    with _silence_stderr():
        return HandLandmarker.create_from_options(options)


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Normalize a (63,) landmark array to be position- and scale-invariant.

    Translates so wrist (index 0) is at the origin, then scales by the
    wrist-to-middle-MCP distance (index 9) so hand size doesn't affect features.
    Returns zeros if the hand span is degenerate.
    """
    pts = lm.reshape(21, 3)
    pts = pts - pts[0]  # translate: wrist to origin
    scale = np.linalg.norm(pts[9])  # wrist-to-middle-MCP = palm span
    if scale > 1e-6:
        pts = pts / scale
    return pts.flatten().astype(np.float32)


def extract_hand_landmarks(image_path, landmarker=None):
    """Extract 63 hand landmark features from a single image.

    Returns a (63,) float32 array of normalized (x, y, z) coords for 21 keypoints,
    or None if no hand is detected.

    Pass a shared landmarker instance when calling in a loop for better performance.
    """
    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = _make_landmarker()

    image = cv2.imread(str(image_path))
    if image is None:
        if owns_landmarker:
            landmarker.close()
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    with _silence_stderr():
        result = landmarker.detect(mp_image)

    if owns_landmarker:
        landmarker.close()

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]
    raw = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
    return normalize_landmarks(raw)


def extract_holistic_landmarks(video_path):
    """Extract per-frame holistic landmarks from a video using MediaPipe Holistic."""
    raise NotImplementedError("Phase 2 - implement holistic landmark extraction")


def _extract_alphabet_landmarks(input_dir, output_dir):
    """Walk input_dir/<class>/*.jpg, extract landmarks, save per-class .npy files."""
    train_dir = os.path.join(input_dir, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.isdir(train_dir):
        # Fallback for flat layout
        train_dir = os.path.join(input_dir, "asl_alphabet_train")
    if not os.path.isdir(train_dir):
        print(f"ERROR: expected '{train_dir}' — run 'make download-alphabet' first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    class_names = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    class_map = {name: idx for idx, name in enumerate(class_names)}
    class_map_path = os.path.join(os.path.dirname(output_dir), "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"Saved class map ({len(class_map)} classes) → {class_map_path}")

    total_extracted = 0
    total_skipped = 0

    # Create one landmarker for the entire batch run
    landmarker = _make_landmarker()

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        landmarks_list = []
        skipped = 0

        for fname in tqdm(image_files, desc=f"{class_name:10s}", leave=False):
            lm = extract_hand_landmarks(os.path.join(class_dir, fname), landmarker=landmarker)
            if lm is None:
                skipped += 1
            else:
                landmarks_list.append(lm)

        if landmarks_list:
            out_path = os.path.join(output_dir, f"{class_name}.npy")
            np.save(out_path, np.stack(landmarks_list))

        total_extracted += len(landmarks_list)
        total_skipped += skipped
        print(f"  {class_name}: {len(landmarks_list)} extracted, {skipped} skipped (no hand detected)")

    landmarker.close()
    print(f"\nDone. Total extracted: {total_extracted}, skipped: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks from images/video")
    parser.add_argument("--dataset", choices=["alphabet", "wlasl"], required=True)
    parser.add_argument("--input-dir", type=str, default="data/asl_alphabet")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "alphabet":
        out = args.output_dir or os.path.join(args.input_dir, "landmarks")
        _extract_alphabet_landmarks(args.input_dir, out)
    elif args.dataset == "wlasl":
        raise NotImplementedError("Phase 2 - WLASL landmark extraction not yet implemented")
