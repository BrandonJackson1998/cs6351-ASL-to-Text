"""
Keyframe Extractor
Extracts representative keyframe images from an ASL video to tell the story
of the video in fewer frames. Multiple keyframes can be extracted per segment
to capture distinct held hand positions within a single sign.

Strategy: finds velocity *valleys* (moments where the hand is still/held)
rather than peaks. These are the frames where the signer is holding a
meaningful hand shape, producing clear and representative images.

Only frames where a hand is detected are considered.

Output structure:
    segments/<video-name>/keyframes/
        seg0000_frame000045.jpg
        seg0000_frame000062.jpg
        seg0001_frame000110.jpg
        ...
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np

from src.preprocessing.video_segmenter import VideoSegmenter, VideoSegment


@dataclass
class Keyframe:
    """A representative frame extracted from a sign segment."""
    segment_index: int
    frame_index: int
    timestamp: float
    image_path: Optional[str] = None


class KeyframeExtractor:
    """Extracts keyframe images from an ASL video.

    Selects keyframes at acceleration peaks and valleys within each segment.
    - Acceleration peaks: hand just launched into motion (start of gesture phase)
    - Acceleration valleys: hand just finished decelerating (end of gesture phase)

    Together these capture the inflection points that define a sign's structure.
    Only frames where a hand is detected are considered.

    Args:
        segmenter: VideoSegmenter instance (uses defaults if not provided).
        max_keyframes_per_segment: Upper bound on frames per segment.
        min_prominence: Minimum prominence for an acceleration peak/valley to qualify.
        accel_smoothing_window: Window size for smoothing the acceleration signal
            before peak/valley detection. Larger = fewer, broader inflection points.
    """

    def __init__(
        self,
        segmenter: Optional[VideoSegmenter] = None,
        max_keyframes_per_segment: int = 10,
        min_prominence: float = 0.002,
        accel_smoothing_window: int = 7,
    ):
        self.segmenter = segmenter or VideoSegmenter()
        self.max_keyframes_per_segment = max_keyframes_per_segment
        self.min_prominence = min_prominence
        self.accel_smoothing_window = accel_smoothing_window

    def extract(
        self, video_path: str, output_dir: str = "segments"
    ) -> List[Keyframe]:
        """Extract keyframes from a video.

        Args:
            video_path: Path to input video.
            output_dir: Base output directory. Keyframes saved under
                        <output_dir>/<video_name>/keyframes/

        Returns:
            List of Keyframe objects ordered by time.
        """
        segments = self.segmenter.segment(video_path)
        if not segments:
            return []

        velocity, acceleration, hand_detected, _, fps = self.segmenter.get_motion_signal(video_path)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframes_dir = os.path.join(output_dir, video_name, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)

        all_keyframe_indices = []
        for seg in segments:
            indices = self._select_segment_keyframes(seg, velocity, acceleration, hand_detected, fps)
            all_keyframe_indices.append(indices)

        keyframes = self._save_keyframes(
            video_path, segments, all_keyframe_indices, keyframes_dir, fps
        )
        return keyframes

    def _select_segment_keyframes(
        self,
        seg: VideoSegment,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        hand_detected: np.ndarray,
        fps: float,
    ) -> List[int]:
        """Select keyframes at acceleration peaks and valleys within a segment.

        Peaks (positive) = hand launching into motion (start of gesture phase).
        Valleys (negative) = hand finished decelerating (end of gesture phase).
        These inflection points capture the distinct phases of each sign.
        """
        start, end = seg.start_frame, seg.end_frame
        seg_accel_raw = acceleration[start:end]
        seg_hand = hand_detected[start:end]
        n = len(seg_accel_raw)

        if n < 3:
            return []

        # Smooth acceleration internally to collapse noisy oscillations
        kernel = np.ones(self.accel_smoothing_window) / self.accel_smoothing_window
        seg_accel = np.convolve(seg_accel_raw, kernel, mode="same")

        # Find all local peaks and valleys of smoothed acceleration where hand is detected
        candidates = []
        for i in range(1, n - 1):
            if not seg_hand[i]:
                continue

            is_peak = seg_accel[i] > seg_accel[i - 1] and seg_accel[i] > seg_accel[i + 1]
            is_valley = seg_accel[i] < seg_accel[i - 1] and seg_accel[i] < seg_accel[i + 1]

            if is_peak or is_valley:
                prominence = abs(seg_accel[i])
                if prominence >= self.min_prominence:
                    candidates.append((i, prominence))

        # Take all qualifying candidates up to the max, sorted by prominence
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = [c[0] for c in candidates[:self.max_keyframes_per_segment]]
            selected.sort()
            return [start + s for s in selected]

        # Fallback: midpoint of segment if hand is detected there
        mid = n // 2
        if seg_hand[mid]:
            return [start + mid]
        for offset in range(1, n):
            if mid + offset < n and seg_hand[mid + offset]:
                return [start + mid + offset]
            if mid - offset >= 0 and seg_hand[mid - offset]:
                return [start + mid - offset]
        return []

    def _save_keyframes(
        self,
        video_path: str,
        segments: List[VideoSegment],
        keyframe_indices: List[List[int]],
        output_dir: str,
        fps: float,
    ) -> List[Keyframe]:
        """Read specific frames from the video and save as images."""
        cap = cv2.VideoCapture(video_path)
        keyframes = []

        for seg_i, (seg, indices) in enumerate(zip(segments, keyframe_indices)):
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                image_path = os.path.join(
                    output_dir, f"seg{seg_i:04d}_frame{frame_idx:06d}.jpg"
                )
                cv2.imwrite(image_path, frame)

                kf = Keyframe(
                    segment_index=seg_i,
                    frame_index=frame_idx,
                    timestamp=frame_idx / fps,
                    image_path=image_path,
                )
                keyframes.append(kf)

        cap.release()
        return keyframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframe images from ASL video")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="segments/", help="Base output directory")
    parser.add_argument("--max-per-segment", type=int, default=10,
                        help="Max keyframes per segment")
    parser.add_argument("--min-prominence", type=float, default=0.002,
                        help="Min acceleration prominence for a keyframe candidate")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=7)
    args = parser.parse_args()

    segmenter = VideoSegmenter(
        min_segment_duration=args.min_duration,
        smoothing_window=args.smoothing_window,
    )

    extractor = KeyframeExtractor(
        segmenter=segmenter,
        max_keyframes_per_segment=args.max_per_segment,
        min_prominence=args.min_prominence,
    )
    keyframes = extractor.extract(args.input, output_dir=args.output_dir)

    print(f"Extracted {len(keyframes)} keyframes across {len(set(kf.segment_index for kf in keyframes))} segments:")
    current_seg = -1
    for kf in keyframes:
        if kf.segment_index != current_seg:
            current_seg = kf.segment_index
            print(f"  Segment {current_seg}:")
        print(f"    frame {kf.frame_index} ({kf.timestamp:.2f}s) -> {kf.image_path}")
