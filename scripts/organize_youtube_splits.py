"""Organize YouTube clips into train/val/test splits.

Takes extracted landmarks from YouTube clips and organizes them
into the format expected by train_word_sign.py.

Usage:
    python scripts/organize_youtube_splits.py \
        --metadata data/youtube_clips/VIDEO_ID_metadata.json \
        --landmarks-dir data/youtube_landmarks \
        --output-dir data/youtube_wordsigns
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def load_metadata(metadata_path):
    """Load YouTube clip metadata."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata


def organize_splits(metadata, landmarks_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """Organize clips into train/val/test splits."""

    # Group clips by label
    clips_by_label = {}
    for clip in metadata['clips']:
        label = clip['label']
        if label not in clips_by_label:
            clips_by_label[label] = []
        clips_by_label[label].append(clip)

    print(f"Found {len(clips_by_label)} unique labels:")
    for label, clips in clips_by_label.items():
        print(f"  {label}: {len(clips)} samples")

    # For single-sample labels, we can't split properly
    # Put first N% in train, next in val, rest in test
    all_samples = {
        'train': [],
        'val': [],
        'test': []
    }

    # Create vocabulary
    vocabulary = {label: idx for idx, label in enumerate(sorted(clips_by_label.keys()))}

    for label, clips in clips_by_label.items():
        label_idx = vocabulary[label]

        # For now, since we only have 1 sample per label, put all in train
        # In a real scenario with multiple samples, we'd split properly
        for i, clip in enumerate(clips):
            # Construct landmark path
            video_id = metadata['video_id']
            start = clip['start']
            end = clip['end']
            landmark_filename = f"{video_id}_{label}_{start}-{end}.npy"
            landmark_path = os.path.join(landmarks_dir, landmark_filename)

            if not os.path.exists(landmark_path):
                print(f"WARNING: Landmark not found: {landmark_path}")
                continue

            # Determine split (for single sample, put in train)
            if len(clips) == 1:
                split = 'train'
            else:
                # If multiple samples, split them
                n_train = max(1, int(len(clips) * train_ratio))
                n_val = max(1, int(len(clips) * val_ratio))

                if i < n_train:
                    split = 'train'
                elif i < n_train + n_val:
                    split = 'val'
                else:
                    split = 'test'

            all_samples[split].append({
                'label': label_idx,
                'path': landmark_path,
                'sample_id': f"{label}_{i}"
            })

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in all_samples.items():
        if not samples:
            continue

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Copy landmarks to organized structure
        split_samples = []
        for sample in samples:
            label_idx = sample['label']
            label_name = [k for k, v in vocabulary.items() if v == label_idx][0]

            # Create label directory
            label_dir = os.path.join(split_dir, f"class{label_idx:03d}")
            os.makedirs(label_dir, exist_ok=True)

            # Copy landmark file
            src_path = sample['path']
            dst_filename = f"{sample['sample_id']}.npy"
            dst_path = os.path.join(label_dir, dst_filename)

            shutil.copy(src_path, dst_path)

            # Add to metadata
            split_samples.append({
                'sample_id': sample['sample_id'],
                'label': label_idx,
                'path': os.path.join(split_name, f"class{label_idx:03d}", dst_filename)
            })

        # Save split metadata
        metadata_path = os.path.join(output_dir, f'{split_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(split_samples, f, indent=2)

        print(f"  {split_name}: {len(split_samples)} samples → {metadata_path}")

    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocabulary, f, indent=2)

    print(f"\n✓ Vocabulary: {len(vocabulary)} classes → {vocab_path}")

    return vocabulary, all_samples


def main():
    parser = argparse.ArgumentParser(description="Organize YouTube clips into dataset splits")
    parser.add_argument('--metadata', required=True,
                        help="YouTube metadata JSON file")
    parser.add_argument('--landmarks-dir', required=True,
                        help="Directory containing extracted landmarks")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for organized dataset")
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    print("="*60)
    print("Organizing YouTube Clips into Dataset")
    print("="*60)

    # Load metadata
    metadata = load_metadata(args.metadata)
    print(f"Video ID: {metadata['video_id']}")
    print(f"Total clips: {len(metadata['clips'])}")
    print()

    # Organize splits
    vocabulary, samples = organize_splits(
        metadata,
        args.landmarks_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio
    )

    # Summary
    print("\n" + "="*60)
    print("Dataset Created")
    print("="*60)
    print(f"Train samples: {len(samples['train'])}")
    print(f"Val samples: {len(samples['val'])}")
    print(f"Test samples: {len(samples['test'])}")
    print(f"Total: {sum(len(s) for s in samples.values())}")
    print(f"Vocabulary size: {len(vocabulary)}")
    print("="*60)

    print("\nNext step:")
    print(f"  python -m src.models.train_word_sign --data-dir {args.output_dir} --experiment youtube_phrases")


if __name__ == "__main__":
    main()
