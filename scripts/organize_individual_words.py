"""Organize individual word landmarks into train/val/test splits.

Groups samples by word label and creates proper dataset structure.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_landmarks_by_word(landmarks_dir):
    """Load all landmark files grouped by word."""
    words = defaultdict(list)

    for npy_file in Path(landmarks_dir).glob("*.npy"):
        filename = npy_file.stem  # Remove .npy extension

        # Parse filename: 3or8QRuQXhI_WORD_NUM_START-END
        parts = filename.split('_')
        if len(parts) < 3:
            continue

        # Extract word label (everything between video ID and number)
        # e.g., "3or8QRuQXhI_YOU_1_32-35" → "YOU"
        word = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1]

        words[word].append(str(npy_file))

    return words


def create_splits(words_dict, train_ratio=0.7, val_ratio=0.15, min_samples=2):
    """Create train/val/test splits from word samples."""

    # Filter words with minimum samples
    filtered_words = {word: samples for word, samples in words_dict.items()
                      if len(samples) >= min_samples}

    print(f"Words with {min_samples}+ samples: {len(filtered_words)}/{len(words_dict)}")
    for word, samples in filtered_words.items():
        print(f"  {word}: {len(samples)} samples")

    # Create vocabulary
    vocabulary = {word: idx for idx, word in enumerate(sorted(filtered_words.keys()))}

    # Create splits
    splits = {'train': [], 'val': [], 'test': []}

    for word, sample_paths in filtered_words.items():
        label_idx = vocabulary[word]
        n_samples = len(sample_paths)

        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(0, int(n_samples * val_ratio))

        for i, path in enumerate(sample_paths):
            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'

            splits[split].append({
                'word': word,
                'label': label_idx,
                'path': path,
                'sample_id': f"{word}_{i}"
            })

    return vocabulary, splits


def organize_dataset(vocabulary, splits, output_dir):
    """Organize files into dataset structure."""

    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in splits.items():
        if not samples:
            # Create empty split file
            with open(os.path.join(output_dir, f'{split_name}.json'), 'w') as f:
                json.dump([], f)
            continue

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_metadata = []

        for sample in samples:
            label_idx = sample['label']
            word = sample['word']

            # Create class directory
            class_dir = os.path.join(split_dir, f"class{label_idx:03d}")
            os.makedirs(class_dir, exist_ok=True)

            # Copy landmark file
            src_path = sample['path']
            dst_filename = f"{sample['sample_id']}.npy"
            dst_path = os.path.join(class_dir, dst_filename)

            shutil.copy(src_path, dst_path)

            # Add to metadata
            split_metadata.append({
                'sample_id': sample['sample_id'],
                'label': label_idx,
                'path': os.path.join(split_name, f"class{label_idx:03d}", dst_filename)
            })

        # Save split metadata
        metadata_path = os.path.join(output_dir, f'{split_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        print(f"  {split_name}: {len(split_metadata)} samples")

    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocabulary, f, indent=2)

    print(f"\n✓ Vocabulary: {len(vocabulary)} words")


def main():
    parser = argparse.ArgumentParser(description="Organize individual word dataset")
    parser.add_argument('--landmarks-dir', required=True,
                        help="Directory with landmark .npy files")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for organized dataset")
    parser.add_argument('--min-samples', type=int, default=2,
                        help="Minimum samples required per word")
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    print("="*60)
    print("Organizing Individual Word Dataset")
    print("="*60)

    # Load landmarks by word
    words_dict = load_landmarks_by_word(args.landmarks_dir)
    print(f"\nFound {len(words_dict)} unique words")
    print(f"Total samples: {sum(len(samples) for samples in words_dict.values())}")

    # Create splits
    vocabulary, splits = create_splits(
        words_dict,
        args.train_ratio,
        args.val_ratio,
        args.min_samples
    )

    print(f"\nSplit summary:")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")

    # Organize files
    print(f"\nOrganizing files to {args.output_dir}...")
    organize_dataset(vocabulary, splits, args.output_dir)

    print("\n" + "="*60)
    print("Dataset Ready")
    print("="*60)
    print(f"Next: python -m src.models.train_word_sign --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
