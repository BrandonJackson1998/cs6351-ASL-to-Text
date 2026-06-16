#!/usr/bin/env python3
"""
Extract hand crop images from existing landmark datasets.

Processes either:
1. Kaggle ASL Alphabet (static images with precomputed landmarks)
2. Balanced letter frames (mixed source with precomputed landmarks)

Creates hand-cropped 200x200 images suitable for CNN training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.hand_crop_extractor import extract_hand_crop


def process_kaggle_alphabet(input_dir: Path, output_dir: Path, target_size=(200, 200)):
    """Process Kaggle ASL Alphabet dataset.

    Structure:
        input_dir/
            asl_alphabet/
                landmarks/
                    A.npy  # (N, 63) landmarks
                    ...
                asl_alphabet_train/
                    A/
                        *.jpg  # source images
                    ...
    """
    landmarks_dir = input_dir / "asl_alphabet" / "landmarks"
    images_dir = input_dir / "asl_alphabet" / "asl_alphabet_train" / "asl_alphabet_train"

    if not landmarks_dir.exists():
        raise FileNotFoundError(f"Landmarks not found: {landmarks_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images not found: {images_dir}")

    print(f"Processing Kaggle ASL Alphabet")
    print(f"  Input landmarks: {landmarks_dir}")
    print(f"  Input images: {images_dir}")
    print(f"  Output: {output_dir}")

    letters = sorted([f.stem for f in landmarks_dir.glob("*.npy")])
    total_extracted = 0
    total_failed = 0

    for letter in letters:
        # Load landmarks
        landmark_file = landmarks_dir / f"{letter}.npy"
        landmarks = np.load(landmark_file)

        # Get corresponding images
        letter_img_dir = images_dir / letter
        if not letter_img_dir.exists():
            print(f"  Warning: No images for {letter}")
            continue

        image_files = sorted(letter_img_dir.glob("*.jpg"))

        if len(image_files) != len(landmarks):
            print(f"  Warning: {letter} has {len(image_files)} images but {len(landmarks)} landmarks")
            # Take minimum
            n = min(len(image_files), len(landmarks))
            image_files = image_files[:n]
            landmarks = landmarks[:n]

        # Create output directory
        output_letter_dir = output_dir / letter
        output_letter_dir.mkdir(parents=True, exist_ok=True)

        # Process each image
        extracted = 0
        failed = 0

        for i, img_path in enumerate(tqdm(image_files, desc=f"  {letter}", leave=False)):
            # Load image
            frame = cv2.imread(str(img_path))
            if frame is None:
                failed += 1
                continue

            # Extract crop
            crop = extract_hand_crop(frame, landmarks[i], target_size=target_size)

            if crop is not None:
                # Save as RGB
                output_path = output_letter_dir / f"{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                extracted += 1
            else:
                failed += 1

        total_extracted += extracted
        total_failed += failed

        print(f"  {letter}: {extracted} extracted, {failed} failed")

    print(f"\nTotal: {total_extracted} extracted, {total_failed} failed")
    return total_extracted, total_failed


def process_balanced_frames(data_dir: Path, output_dir: Path, target_size=(200, 200)):
    """Process balanced_letter_frames dataset.

    This dataset has landmarks but the source frames come from multiple videos.
    We'll need to trace back to the source data.

    For now, we'll skip this and focus on Kaggle first.
    """
    print("ERROR: balanced_letter_frames requires video source tracking")
    print("Use Kaggle dataset for now with --dataset kaggle")
    return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand crop images from landmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Kaggle ASL Alphabet
  python scripts/extract_hand_crops.py \\
      --dataset kaggle \\
      --input data \\
      --output data/hand_crops_kaggle

  # Custom crop size
  python scripts/extract_hand_crops.py \\
      --dataset kaggle \\
      --input data \\
      --output data/hand_crops_kaggle_224 \\
      --size 224
"""
    )

    parser.add_argument(
        "--dataset",
        choices=["kaggle", "balanced"],
        required=True,
        help="Dataset type to process"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input data directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for hand crops"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Crop size (square, default: 200)"
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Padding around hand bbox (default: 0.2 = 20%%)"
    )

    args = parser.parse_args()

    target_size = (args.size, args.size)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "input_dir": str(args.input),
        "target_size": target_size,
        "padding": args.padding,
    }

    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Process dataset
    if args.dataset == "kaggle":
        extracted, failed = process_kaggle_alphabet(
            args.input, args.output, target_size
        )
    elif args.dataset == "balanced":
        extracted, failed = process_balanced_frames(
            args.input, args.output, target_size
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"Extracted: {extracted}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*extracted/(extracted+failed):.1f}%")
    print(f"Output: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
