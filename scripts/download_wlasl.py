"""Download and prepare WLASL dataset for word-sign recognition.

WLASL (Word-Level American Sign Language) dataset:
- WLASL100: 100 most common signs
- WLASL300: 300 common signs
- WLASL2000: 2000 signs (full vocabulary)

Dataset structure:
    data/wlasl/
        videos/         Raw videos (downloaded from source)
        landmarks/      Extracted MediaPipe landmarks
        splits/         Train/val/test splits
        WLASL_v0.3.json  Metadata

Reference: https://dxli94.github.io/WLASL/
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def download_wlasl_json(output_dir):
    """Download WLASL metadata JSON."""
    url = "https://github.com/dxli94/WLASL/raw/master/WLASL_v0.3.json"
    output_path = os.path.join(output_dir, "WLASL_v0.3.json")

    if os.path.exists(output_path):
        print(f"✓ WLASL metadata already exists: {output_path}")
        return output_path

    print(f"Downloading WLASL metadata from {url}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"ERROR: Failed to download WLASL metadata: {e}")
        sys.exit(1)


def load_wlasl_metadata(json_path):
    """Load WLASL JSON and return filtered vocabulary."""
    with open(json_path) as f:
        data = json.load(f)

    print(f"Loaded WLASL metadata: {len(data)} words total")
    return data


def filter_vocabulary(data, subset='100'):
    """Filter WLASL to specific vocabulary size.

    Args:
        data: Full WLASL metadata list
        subset: '100', '300', or '2000'

    Returns:
        Filtered list of word entries
    """
    subset_size = int(subset)

    # WLASL metadata has an 'instances' field with video URLs per word
    # Filter to words with sufficient video samples
    filtered = []
    for entry in data:
        if len(filtered) >= subset_size:
            break
        if len(entry.get('instances', [])) >= 3:  # At least 3 videos per word
            filtered.append(entry)

    print(f"Filtered to WLASL{subset}: {len(filtered)} words")
    return filtered


def create_vocabulary_map(filtered_data):
    """Create word → class_id mapping."""
    vocab = {entry['gloss']: idx for idx, entry in enumerate(filtered_data)}
    return vocab


def create_splits(filtered_data, split_ratios=(0.7, 0.15, 0.15)):
    """Create train/val/test splits per word.

    Returns:
        splits: dict with 'train', 'val', 'test' keys
                Each value is list of (word, video_id, url) tuples
    """
    train_split, val_split, test_split = [], [], []

    for entry in filtered_data:
        word = entry['gloss']
        instances = entry['instances']

        # Shuffle instances deterministically
        np.random.seed(42)
        indices = np.random.permutation(len(instances))

        # Split
        n_train = int(len(instances) * split_ratios[0])
        n_val = int(len(instances) * split_ratios[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        for idx in train_idx:
            inst = instances[idx]
            train_split.append({
                'word': word,
                'video_id': inst['video_id'],
                'url': inst.get('url', ''),
                'bbox': inst.get('bbox', []),
                'frame_start': inst.get('frame_start', -1),
                'frame_end': inst.get('frame_end', -1),
            })

        for idx in val_idx:
            inst = instances[idx]
            val_split.append({
                'word': word,
                'video_id': inst['video_id'],
                'url': inst.get('url', ''),
                'bbox': inst.get('bbox', []),
                'frame_start': inst.get('frame_start', -1),
                'frame_end': inst.get('frame_end', -1),
            })

        for idx in test_idx:
            inst = instances[idx]
            test_split.append({
                'word': word,
                'video_id': inst['video_id'],
                'url': inst.get('url', ''),
                'bbox': inst.get('bbox', []),
                'frame_start': inst.get('frame_start', -1),
                'frame_end': inst.get('frame_end', -1),
            })

    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


def save_splits(splits, vocab, output_dir):
    """Save splits and vocabulary to disk."""
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # Save vocabulary
    vocab_path = os.path.join(splits_dir, 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"✓ Saved vocabulary ({len(vocab)} words) to {vocab_path}")

    # Save splits
    for split_name, split_data in splits.items():
        split_path = os.path.join(splits_dir, f'{split_name}.json')
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"✓ Saved {split_name} split ({len(split_data)} samples) to {split_path}")


def print_statistics(splits, vocab):
    """Print dataset statistics."""
    print("\n" + "="*60)
    print("WLASL Dataset Statistics")
    print("="*60)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Total samples: {sum(len(s) for s in splits.values())}")

    # Samples per word
    word_counts = {}
    for split_data in splits.values():
        for sample in split_data:
            word = sample['word']
            word_counts[word] = word_counts.get(word, 0) + 1

    print(f"\nSamples per word: {min(word_counts.values())} - {max(word_counts.values())}")
    print(f"Average: {np.mean(list(word_counts.values())):.1f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare WLASL dataset")
    parser.add_argument('--output-dir', default='data/wlasl',
                        help="Output directory for WLASL data")
    parser.add_argument('--subset', choices=['100', '300', '2000'], default='100',
                        help="Vocabulary size (WLASL100, WLASL300, or WLASL2000)")
    args = parser.parse_args()

    print("="*60)
    print(f"Preparing WLASL{args.subset} dataset")
    print("="*60)

    # Step 1: Download metadata
    json_path = download_wlasl_json(args.output_dir)

    # Step 2: Load and filter vocabulary
    full_data = load_wlasl_metadata(json_path)
    filtered_data = filter_vocabulary(full_data, args.subset)

    # Step 3: Create vocabulary mapping
    vocab = create_vocabulary_map(filtered_data)

    # Step 4: Create train/val/test splits
    splits = create_splits(filtered_data)

    # Step 5: Save splits and vocabulary
    save_splits(splits, vocab, args.output_dir)

    # Step 6: Print statistics
    print_statistics(splits, vocab)

    print("\n" + "="*60)
    print("✓ WLASL metadata prepared successfully!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Download videos: make download-wlasl-videos SUBSET={args.subset}")
    print(f"2. Extract landmarks: make extract-wlasl-landmarks SUBSET={args.subset}")
    print(f"3. Train model: make train-word-sign EXP=wlasl{args.subset}")


if __name__ == "__main__":
    main()
