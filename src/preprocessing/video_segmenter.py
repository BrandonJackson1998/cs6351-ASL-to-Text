"""
Video Segmenter
Segments an ASL video into individual sign clips using adaptive landmark heuristics.

Uses the video's own signal statistics to determine boundaries rather than
hardcoded thresholds. Between consecutive signs, signers produce velocity
valleys -- the segmenter finds these by looking for valleys whose depth
(prominence) is significant relative to the surrounding signal.

Signals used:
    1. Hand velocity (wrist displacement between frames)
    2. Acceleration (derivative of velocity)
    3. Hand presence (prolonged gaps only, to avoid MediaPipe dropout noise)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import contextlib
import cv2
import numpy as np

# Set MEDIAPIPE_VERBOSE=1 to see MediaPipe's C-level logs (suppressed by default).
_VERBOSE = os.environ.get("MEDIAPIPE_VERBOSE", "0") == "1"

@contextlib.contextmanager
def _silence_stderr():
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

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hand_landmarker.task")


def _download_model():
    """Download MediaPipe hand_landmarker model if not present."""
    import sys
    import subprocess

    model_path = os.path.abspath(_MODEL_PATH)
    model_dir = os.path.dirname(model_path)

    print(f"MediaPipe model not found at '{model_path}'")
    print("Downloading hand_landmarker.task (7.5MB) from Google MediaPipe...")

    os.makedirs(model_dir, exist_ok=True)

    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    # Try curl first (more reliable on macOS)
    try:
        subprocess.run(
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


def _make_video_landmarker():
    model_path = os.path.abspath(_MODEL_PATH)
    if not os.path.isfile(model_path):
        _download_model()
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with _silence_stderr():
        return HandLandmarker.create_from_options(options)


@dataclass
class VideoSegment:
    """A single sign segment extracted from a video."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    clip_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class FrameLandmarks:
    """Per-frame landmark data extracted from MediaPipe."""
    frame_idx: int
    hand_detected: bool
    # Wrist position (normalized 0-1, origin top-left)
    wrist_x: float = 0.0
    wrist_y: float = 0.0
    # If two hands detected, use the dominant (right) or average
    wrist_x_left: Optional[float] = None
    wrist_y_left: Optional[float] = None


class VideoSegmenter:
    """Segments ASL video into individual sign clips using adaptive heuristics.

    Finds boundaries by detecting prominent velocity valleys in the signal.
    Thresholds are derived from the video's own statistics rather than
    hardcoded values, so the segmenter adapts to different signing speeds,
    camera distances, and video qualities.

    Args:
        min_segment_duration: Minimum clip length in seconds.
        valley_percentile: Velocity values below this percentile of the
            signal are considered "low". Default 25 = bottom quartile.
        min_prominence_ratio: A valley must drop this fraction of the signal's
            dynamic range below its surrounding peaks to count as a boundary.
            Higher = fewer, more distinct boundaries.
        min_gap_duration: Minimum hand-detection gap (seconds) to count as a
            boundary. Prevents brief MediaPipe dropouts from splitting signs.
        smoothing_window: Rolling average window size for velocity smoothing.
    """

    def __init__(
        self,
        min_segment_duration: float = 0.5,
        valley_percentile: float = 25.0,
        min_prominence_ratio: float = 0.3,
        min_gap_duration: float = 0.3,
        smoothing_window: int = 7,
    ):
        self.min_segment_duration = min_segment_duration
        self.valley_percentile = valley_percentile
        self.min_prominence_ratio = min_prominence_ratio
        self.min_gap_duration = min_gap_duration
        self.smoothing_window = smoothing_window

    def segment(self, video_path: str, output_dir: Optional[str] = None) -> List[VideoSegment]:
        """Segment a video into individual sign clips.

        Args:
            video_path: Path to the input video file.
            output_dir: If provided, save each clip as a separate video file.

        Returns:
            List of VideoSegment objects describing each detected sign.
        """
        frame_landmarks = self._extract_frame_landmarks(video_path)
        if not frame_landmarks:
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps <= 0:
            raise ValueError(f"Could not read FPS from video: {video_path}")

        velocity, acceleration, hand_detected, wrist_y = self._compute_motion_signal(frame_landmarks)
        boundaries = self._find_boundaries(velocity, acceleration, hand_detected, fps)

        # Convert boundaries to segment ranges
        segment_frames = self._boundaries_to_segments(boundaries, total_frames, fps)

        # Build VideoSegment objects
        segments = []
        for i, (start, end) in enumerate(segment_frames):
            seg = VideoSegment(
                start_frame=start,
                end_frame=end,
                start_time=start / fps,
                end_time=end / fps,
            )
            segments.append(seg)

        # Optionally extract clips to disk
        if output_dir and segments:
            self._extract_clips(video_path, segments, output_dir)

        return segments

    def _extract_frame_landmarks(self, video_path: str) -> list[FrameLandmarks]:
        """Extract per-frame wrist landmarks using MediaPipe HandLandmarker (Tasks API)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        landmarker = _make_video_landmarker()
        frame_landmarks = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            with _silence_stderr():
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks and results.handedness:
                right_wrist = None
                left_wrist = None

                for hand_lm, handedness in zip(results.hand_landmarks, results.handedness):
                    label = handedness[0].category_name  # "Left" or "Right"
                    wrist = hand_lm[0]  # landmark index 0 = WRIST

                    if label == "Right":
                        right_wrist = (wrist.x, wrist.y)
                    else:
                        left_wrist = (wrist.x, wrist.y)

                primary = right_wrist or left_wrist
                fl = FrameLandmarks(
                    frame_idx=frame_idx,
                    hand_detected=True,
                    wrist_x=primary[0],
                    wrist_y=primary[1],
                )
                if left_wrist and right_wrist:
                    fl.wrist_x_left = left_wrist[0]
                    fl.wrist_y_left = left_wrist[1]

                frame_landmarks.append(fl)
            else:
                frame_landmarks.append(FrameLandmarks(frame_idx=frame_idx, hand_detected=False))

            frame_idx += 1

        cap.release()
        landmarker.close()
        return frame_landmarks

    def _compute_motion_signal(
        self, landmarks: list[FrameLandmarks]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute smoothed velocity, acceleration, hand_detected, and wrist_y signals.

        Returns:
            velocity: Smoothed wrist displacement per frame (length N).
            acceleration: Smoothed rate of change of velocity per frame (length N).
            hand_detected: Boolean array (length N).
            wrist_y: Wrist y-position per frame (length N), 0 where no hand.
        """
        n = len(landmarks)
        raw_velocity = np.zeros(n)
        hand_detected = np.array([fl.hand_detected for fl in landmarks])
        wrist_y = np.array([fl.wrist_y for fl in landmarks])

        for i in range(1, n):
            if landmarks[i].hand_detected and landmarks[i - 1].hand_detected:
                dx = landmarks[i].wrist_x - landmarks[i - 1].wrist_x
                dy = landmarks[i].wrist_y - landmarks[i - 1].wrist_y
                raw_velocity[i] = np.sqrt(dx**2 + dy**2)
            else:
                raw_velocity[i] = 0.0

        # Smooth velocity with rolling average
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        velocity = np.convolve(raw_velocity, kernel, mode="same")

        # Acceleration = derivative of velocity (change in velocity per frame)
        raw_acceleration = np.diff(velocity, prepend=velocity[0])
        acceleration = np.convolve(raw_acceleration, kernel, mode="same")

        return velocity, acceleration, hand_detected, wrist_y

    def _find_boundaries(
        self,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        hand_detected: np.ndarray,
        fps: float,
    ) -> list[int]:
        """Detect sign boundaries adaptively from the velocity signal.

        1. Compute the signal's own statistics (percentiles, dynamic range).
        2. Find all local velocity minima (valleys).
        3. Compute each valley's prominence (how much it dips below its
           surrounding peaks).
        4. Keep only valleys whose prominence exceeds a fraction of the
           signal's dynamic range.
        5. Add boundaries for prolonged hand-detection gaps.
        6. Enforce minimum segment duration by merging close boundaries.
        """
        n = len(velocity)
        min_seg_frames = max(1, int(self.min_segment_duration * fps))
        min_gap_frames = max(1, int(self.min_gap_duration * fps))

        # --- Adaptive thresholds from signal statistics ---
        vel_active = velocity[hand_detected.astype(bool)]
        if len(vel_active) == 0:
            vel_active = velocity

        vel_floor = np.percentile(vel_active, self.valley_percentile)
        dynamic_range = np.percentile(vel_active, 95) - np.percentile(vel_active, 5)
        min_prominence = self.min_prominence_ratio * dynamic_range

        # --- Signal 1: Prominent velocity valleys ---
        # Find all local minima
        valleys = []
        for i in range(1, n - 1):
            if velocity[i] <= velocity[i - 1] and velocity[i] <= velocity[i + 1]:
                if velocity[i] <= vel_floor:
                    valleys.append(i)

        # Compute prominence for each valley:
        # how far it dips below the lower of the two nearest peaks on each side
        prominent_valleys = []
        for v in valleys:
            # Find nearest peak to the left
            left_peak = velocity[v]
            for j in range(v - 1, -1, -1):
                if velocity[j] > velocity[j + 1]:
                    left_peak = velocity[j]
                    break
                left_peak = max(left_peak, velocity[j])

            # Find nearest peak to the right
            right_peak = velocity[v]
            for j in range(v + 1, n):
                if velocity[j] > velocity[j - 1]:
                    right_peak = velocity[j]
                    break
                right_peak = max(right_peak, velocity[j])

            prominence = min(left_peak, right_peak) - velocity[v]
            if prominence >= min_prominence:
                prominent_valleys.append((v, prominence))

        # Sort by prominence descending, greedily select with spacing
        prominent_valleys.sort(key=lambda x: x[1], reverse=True)
        boundaries = []
        for v_idx, _ in prominent_valleys:
            if all(abs(v_idx - b) >= min_seg_frames for b in boundaries):
                boundaries.append(v_idx)

        # --- Signal 2: Prolonged hand-detection gaps ---
        gap_start = None
        for i in range(n):
            if not hand_detected[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None and (i - gap_start) >= min_gap_frames:
                    center = (gap_start + i) // 2
                    if all(abs(center - b) >= min_seg_frames for b in boundaries):
                        boundaries.append(center)
                gap_start = None
        if gap_start is not None and (n - gap_start) >= min_gap_frames:
            center = (gap_start + n) // 2
            if all(abs(center - b) >= min_seg_frames for b in boundaries):
                boundaries.append(center)

        boundaries.sort()
        return boundaries

    def _boundaries_to_segments(
        self, boundaries: list[int], total_frames: int, fps: float
    ) -> list[tuple[int, int]]:
        """Convert boundary frame indices to (start, end) segment pairs.

        Filters out segments shorter than min_segment_duration.
        """
        min_frames = int(self.min_segment_duration * fps)

        # Build segment ranges from boundaries
        edges = [0] + boundaries + [total_frames]
        segments = []

        for i in range(len(edges) - 1):
            start = edges[i]
            end = edges[i + 1]
            if (end - start) >= min_frames:
                segments.append((start, end))

        return segments

    def _extract_clips(
        self, video_path: str, segments: list[VideoSegment], output_dir: str
    ) -> None:
        """Save each segment as a separate video file in a per-video subdirectory."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clips_dir = os.path.join(output_dir, video_name, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for i, seg in enumerate(segments):
            clip_path = os.path.join(clips_dir, f"segment_{i:04d}.mp4")
            writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
            for _ in range(seg.frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)

            writer.release()
            seg.clip_path = clip_path

        cap.release()

    def get_motion_signal(self, video_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Public access to motion signal for visualization/debugging.

        Returns:
            velocity, acceleration, hand_detected, wrist_y, fps
        """
        frame_landmarks = self._extract_frame_landmarks(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        velocity, acceleration, hand_detected, wrist_y = self._compute_motion_signal(frame_landmarks)
        return velocity, acceleration, hand_detected, wrist_y, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment ASL video into individual sign clips")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="segments/", help="Directory to save clips")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Min segment duration (seconds)")
    parser.add_argument("--valley-percentile", type=float, default=25.0, help="Velocity percentile for valley threshold")
    parser.add_argument("--min-prominence-ratio", type=float, default=0.3, help="Min prominence as fraction of dynamic range")
    parser.add_argument("--min-gap-duration", type=float, default=0.3, help="Min hand-detection gap (seconds)")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--visualize", action="store_true", help="Show segmentation visualization")
    args = parser.parse_args()

    segmenter = VideoSegmenter(
        min_segment_duration=args.min_duration,
        valley_percentile=args.valley_percentile,
        min_prominence_ratio=args.min_prominence_ratio,
        min_gap_duration=args.min_gap_duration,
        smoothing_window=args.smoothing_window,
    )

    segments = segmenter.segment(args.input, output_dir=args.output_dir)

    print(f"Found {len(segments)} segments:")
    for i, seg in enumerate(segments):
        print(f"  [{i}] frames {seg.start_frame}-{seg.end_frame} "
              f"({seg.start_time:.2f}s - {seg.end_time:.2f}s, {seg.duration:.2f}s)")

    if args.visualize:
        from scripts.visualize_segmentation import plot_segmentation
        velocity, acceleration, hand_detected, wrist_y, fps = segmenter.get_motion_signal(args.input)
        boundaries = [seg.start_frame for seg in segments[1:]]
        plot_segmentation(velocity, acceleration, hand_detected, wrist_y, boundaries, fps)
