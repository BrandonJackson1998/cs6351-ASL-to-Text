"""Extract individual word signs from conversational phrases.

This script helps identify individual signs within phrase videos and
extracts them as separate clips for training.

For a phrase like "DO YOU WANT COFFEE", we want separate clips for:
- DO
- YOU
- WANT
- COFFEE

Strategy:
1. Load the full phrase video
2. Detect sign boundaries (similar to word segmentation)
3. Extract individual sign clips
4. Label them manually or via metadata
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_wlasl_landmarks import extract_landmarks_from_video


def detect_sign_boundaries(video_path, min_pause_frames=10):
    """Detect boundaries between individual signs in a phrase.

    Uses velocity-based approach: signs have movement, pauses have stillness.
    """
    # Extract landmarks
    landmarks = extract_landmarks_from_video(video_path)
    if landmarks is None:
        return None

    # Calculate frame-to-frame velocity
    velocity = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)

    # Smooth velocity
    window = 5
    velocity_smooth = np.convolve(velocity, np.ones(window)/window, mode='same')

    # Find low-velocity regions (pauses between signs)
    threshold = np.percentile(velocity_smooth, 20)  # Bottom 20% = pauses
    is_pause = velocity_smooth < threshold

    # Find continuous pause regions
    boundaries = []
    in_pause = False
    pause_start = 0

    for i, pause in enumerate(is_pause):
        if pause and not in_pause:
            pause_start = i
            in_pause = True
        elif not pause and in_pause:
            if i - pause_start >= min_pause_frames:
                boundaries.append((pause_start, i))
            in_pause = False

    return boundaries, velocity_smooth


def segment_phrase_into_signs(video_path, expected_word_count):
    """Segment a phrase video into individual sign clips."""
    boundaries, velocity = detect_sign_boundaries(video_path)

    if boundaries is None:
        print(f"ERROR: Could not extract landmarks from {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create segments between boundaries
    segments = []
    prev_end = 0

    for boundary_start, boundary_end in boundaries:
        if prev_end < boundary_start:
            segments.append({
                'start_frame': prev_end,
                'end_frame': boundary_start,
                'start_sec': prev_end / fps,
                'end_sec': boundary_start / fps,
                'duration': (boundary_start - prev_end) / fps
            })
        prev_end = boundary_end

    # Add final segment
    if prev_end < total_frames:
        segments.append({
            'start_frame': prev_end,
            'end_frame': total_frames,
            'start_sec': prev_end / fps,
            'end_sec': total_frames / fps,
            'duration': (total_frames - prev_end) / fps
        })

    return {
        'video': video_path,
        'fps': fps,
        'total_frames': total_frames,
        'segments': segments,
        'expected_words': expected_word_count
    }


def main():
    parser = argparse.ArgumentParser(description="Segment phrases into individual signs")
    parser.add_argument('--input', required=True, help="Input phrase video")
    parser.add_argument('--expected-words', type=int, required=True,
                        help="Expected number of words in phrase")
    parser.add_argument('--output-json', required=True,
                        help="Output JSON with segment info")
    args = parser.parse_args()

    print(f"Analyzing: {args.input}")
    print(f"Expected words: {args.expected_words}")

    result = segment_phrase_into_signs(args.input, args.expected_words)

    if result is None:
        sys.exit(1)

    print(f"\nFound {len(result['segments'])} segments:")
    for i, seg in enumerate(result['segments'], 1):
        print(f"  Segment {i}: {seg['start_sec']:.2f}s - {seg['end_sec']:.2f}s ({seg['duration']:.2f}s)")

    # Save to JSON
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n✓ Saved to {args.output_json}")
    print("\nNext: Review segments and assign word labels")


if __name__ == "__main__":
    main()
