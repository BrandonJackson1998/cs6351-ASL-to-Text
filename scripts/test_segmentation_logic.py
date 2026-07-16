"""Unit test for word segmentation logic."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transcribe_with_segmentation import segment_by_gaps, format_segmented_output


def test_hand_drop_segmentation():
    """Test that hand drops create word boundaries."""
    # Simulate 3 words: FIG (frames 0-29), hand drop (30-44), DATE (45-74), hand drop (75-89), LIME (90-119)
    fps = 30.0

    # Holds: each letter held for 10 frames
    holds = [
        ('F', 0, 10),
        ('I', 10, 20),
        ('G', 20, 30),
        # Gap with hand drop: frames 30-44
        ('D', 45, 55),
        ('A', 55, 65),
        ('T', 65, 75),
        # Gap with hand drop: frames 75-89
        ('L', 90, 100),
        ('I', 100, 110),
        ('M', 110, 120),
        ('E', 120, 130),
    ]

    # Hand present for all frames except gaps
    hand_present = np.ones(130, dtype=bool)
    hand_present[30:45] = False  # First gap
    hand_present[75:90] = False  # Second gap

    pause_threshold_seconds = 0.3  # 9 frames

    segments = segment_by_gaps(holds, hand_present, fps, pause_threshold_seconds)
    output = format_segmented_output(segments)

    print(f"Test 1 - Hand drops:")
    print(f"  Input holds: {len(holds)} holds")
    print(f"  Output: {output}")
    print(f"  Expected: FIG | DATE | LIME")
    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
    print("  ✓ PASS\n")


def test_long_pause_segmentation():
    """Test that long pauses (without hand drop) create word boundaries."""
    fps = 30.0

    # Holds with long pauses but hand always present
    holds = [
        ('F', 0, 10),
        ('I', 10, 20),
        ('G', 20, 30),
        # Long pause: 40 frames (1.33s)
        ('D', 70, 80),
        ('A', 80, 90),
        ('T', 90, 100),
        ('E', 100, 110),
    ]

    hand_present = np.ones(110, dtype=bool)  # Hand always present

    pause_threshold_seconds = 1.0  # 30 frames

    segments = segment_by_gaps(holds, hand_present, fps, pause_threshold_seconds)
    output = format_segmented_output(segments)

    print(f"Test 2 - Long pause:")
    print(f"  Input holds: {len(holds)} holds")
    print(f"  Output: {output}")
    print(f"  Expected: FIG | DATE")
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
    print("  ✓ PASS\n")


def test_no_segmentation():
    """Test continuous signing with no breaks."""
    fps = 30.0

    holds = [
        ('F', 0, 10),
        ('I', 10, 20),
        ('G', 20, 30),
        ('D', 35, 45),  # Short 5-frame gap
        ('A', 45, 55),
    ]

    hand_present = np.ones(55, dtype=bool)

    pause_threshold_seconds = 1.0  # 30 frames

    segments = segment_by_gaps(holds, hand_present, fps, pause_threshold_seconds)
    output = format_segmented_output(segments)

    print(f"Test 3 - No segmentation (continuous):")
    print(f"  Input holds: {len(holds)} holds")
    print(f"  Output: {output}")
    print(f"  Expected: FIGDA (single segment)")
    assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
    print("  ✓ PASS\n")


def test_duplicate_collapsing():
    """Test that consecutive duplicate letters are collapsed within segments."""
    fps = 30.0

    holds = [
        ('F', 0, 10),
        ('F', 10, 20),  # Duplicate F
        ('I', 20, 30),
        ('G', 30, 40),
    ]

    hand_present = np.ones(40, dtype=bool)
    pause_threshold_seconds = 1.0

    segments = segment_by_gaps(holds, hand_present, fps, pause_threshold_seconds)
    output = format_segmented_output(segments)

    print(f"Test 4 - Duplicate collapsing:")
    print(f"  Input holds: {len(holds)} holds (F, F, I, G)")
    print(f"  Output: {output}")
    print(f"  Expected: FIG (not FFIG)")
    assert output == "FIG", f"Expected 'FIG', got '{output}'"
    print("  ✓ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing word segmentation logic")
    print("=" * 60 + "\n")

    test_hand_drop_segmentation()
    test_long_pause_segmentation()
    test_no_segmentation()
    test_duplicate_collapsing()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
