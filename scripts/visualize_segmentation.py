"""
Segmentation Visualization
Plots the motion signals with detected segment boundaries and keyframe
positions for debugging/tuning.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


def plot_segmentation(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    hand_detected: np.ndarray,
    wrist_y: np.ndarray,
    boundaries: list,
    fps: float,
    keyframe_frames: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot velocity, acceleration, wrist y-position, hand detection, boundaries, and keyframes."""
    n = len(velocity)
    time_axis = np.arange(n) / fps

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for ax in axes:
        for b in boundaries:
            ax.axvline(x=b / fps, color="red", linestyle="--", alpha=0.7, linewidth=1)
        if keyframe_frames:
            for kf in keyframe_frames:
                ax.axvline(x=kf / fps, color="dodgerblue", linestyle="--", alpha=0.6, linewidth=0.8)

    # Plot 1: Velocity
    ax1 = axes[0]
    ax1.plot(time_axis, velocity, color="steelblue", linewidth=0.8)
    ax1.set_ylabel("Wrist Velocity")
    ax1.set_title("ASL Video Segmentation Analysis")

    # Plot 2: Acceleration
    ax2 = axes[1]
    ax2.plot(time_axis, acceleration, color="mediumseagreen", linewidth=0.8)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)
    ax2.set_ylabel("Acceleration")

    # Plot 3: Wrist Y position
    ax3 = axes[2]
    ax3.plot(time_axis, wrist_y, color="darkorange", linewidth=0.8)
    ax3.set_ylabel("Wrist Y Position")
    ax3.invert_yaxis()

    # Plot 4: Hand detection
    ax4 = axes[3]
    ax4.fill_between(time_axis, hand_detected.astype(float), alpha=0.3, color="green")
    ax4.set_ylabel("Hand Detected")
    ax4.set_xlabel("Time (seconds)")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["No", "Yes"])

    # Legend
    seg_line = mlines.Line2D([], [], color="red", linestyle="--", linewidth=1, label="Segment Boundary")
    legend_handles = [seg_line]
    if keyframe_frames:
        kf_line = mlines.Line2D([], [], color="dodgerblue", linestyle="--", linewidth=0.8, label="Keyframe")
        legend_handles.append(kf_line)
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved segmentation plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ASL video segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file. Defaults to segments/<video>/segmentation_analysis.png")
    parser.add_argument("--output-dir", type=str, default="segments/")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--smoothing-window", type=int, default=7)
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from src.preprocessing.video_segmenter import VideoSegmenter
    from src.preprocessing.keyframe_extractor import KeyframeExtractor

    segmenter = VideoSegmenter(
        min_segment_duration=args.min_duration,
        smoothing_window=args.smoothing_window,
    )

    segments = segmenter.segment(args.input)
    velocity, acceleration, hand_detected, wrist_y, fps = segmenter.get_motion_signal(args.input)
    boundaries = [seg.start_frame for seg in segments[1:]]

    # Extract keyframe positions
    extractor = KeyframeExtractor(segmenter=segmenter)
    keyframes = extractor.extract(args.input, output_dir=args.output_dir)
    keyframe_frames = [kf.frame_index for kf in keyframes]

    save_path = args.save
    if not save_path:
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        video_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, "segmentation_analysis.png")

    print(f"Found {len(segments)} segments, {len(keyframe_frames)} keyframes")
    plot_segmentation(velocity, acceleration, hand_detected, wrist_y, boundaries, fps,
                      keyframe_frames=keyframe_frames, save_path=save_path)
