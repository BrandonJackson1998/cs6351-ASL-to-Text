"""
Landmark Extractor
Extracts hand/pose landmarks from images and video using MediaPipe.

Phase 1: MediaPipe Hands (21 keypoints x 3 coords = 63 features) from static images.
Phase 2: MediaPipe Holistic (hands + pose + face) from video frames.
"""

import argparse


def extract_hand_landmarks(image_path):
    """Extract 63 hand landmark features from a single image using MediaPipe Hands."""
    raise NotImplementedError("Phase 1 - implement hand landmark extraction")


def extract_holistic_landmarks(video_path):
    """Extract per-frame holistic landmarks from a video using MediaPipe Holistic."""
    raise NotImplementedError("Phase 2 - implement holistic landmark extraction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract landmarks from images/video")
    parser.add_argument("--dataset", choices=["alphabet", "wlasl"], required=True)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
