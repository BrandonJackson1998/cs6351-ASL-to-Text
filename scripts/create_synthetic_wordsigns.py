"""Generate synthetic word-sign data for pipeline validation.

Creates synthetic landmark sequences that mimic real word signs:
- Each class has distinct temporal patterns
- Variable sequence lengths
- Realistic noise levels

Usage:
    python scripts/create_synthetic_wordsigns.py --num-classes 20 --samples-per-class 50
"""

import argparse
import json
import os
import numpy as np


def generate_class_template(class_id, input_dim=63, base_seq_len=50):
    """Generate a template landmark sequence for a class."""
    np.random.seed(class_id * 100)

    # Create distinct pattern per class
    # Use sinusoidal patterns with class-specific frequencies
    t = np.linspace(0, 2 * np.pi, base_seq_len)

    template = np.zeros((base_seq_len, input_dim))
    for dim in range(input_dim):
        freq = 1 + (class_id % 5) + (dim % 3) * 0.5
        phase = (class_id * dim) % (2 * np.pi)
        amplitude = 0.3 + 0.2 * (class_id % 3)

        template[:, dim] = amplitude * np.sin(freq * t + phase)

    return template


def augment_template(template, noise_level=0.1):
    """Add variation to template."""
    seq_len, input_dim = template.shape

    # Time warping (stretch/compress)
    warp_factor = np.random.uniform(0.8, 1.2)
    new_len = int(seq_len * warp_factor)

    # Interpolate to new length
    indices = np.linspace(0, seq_len - 1, new_len)
    augmented = np.zeros((new_len, input_dim))

    for dim in range(input_dim):
        augmented[:, dim] = np.interp(indices, np.arange(seq_len), template[:, dim])

    # Add Gaussian noise
    noise = np.random.randn(*augmented.shape) * noise_level
    augmented += noise

    # Random amplitude scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale

    return augmented.astype(np.float32)


def create_dataset(num_classes, samples_per_class, output_dir, input_dim=63, base_seq_len=50):
    """Create synthetic dataset with train/val/test splits."""

    print(f"Creating synthetic dataset: {num_classes} classes, {samples_per_class} samples/class")

    # Create class templates
    templates = {}
    for class_id in range(num_classes):
        templates[class_id] = generate_class_template(class_id, input_dim, base_seq_len)

    # Generate samples
    splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    split_samples = {
        'train': [],
        'val': [],
        'test': []
    }

    for class_id in range(num_classes):
        template = templates[class_id]

        for sample_idx in range(samples_per_class):
            # Augment template
            landmarks = augment_template(template)

            # Determine split
            if sample_idx < samples_per_class * splits['train']:
                split = 'train'
            elif sample_idx < samples_per_class * (splits['train'] + splits['val']):
                split = 'val'
            else:
                split = 'test'

            split_samples[split].append({
                'landmarks': landmarks,
                'label': class_id,
                'class_id': class_id,
                'sample_id': f'class{class_id:03d}_sample{sample_idx:03d}'
            })

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)

    for split_name, samples in split_samples.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save individual .npy files
        for sample in samples:
            landmarks = sample['landmarks']
            class_id = sample['class_id']
            sample_id = sample['sample_id']

            class_dir = os.path.join(split_dir, f'class{class_id:03d}')
            os.makedirs(class_dir, exist_ok=True)

            landmark_path = os.path.join(class_dir, f'{sample_id}.npy')
            np.save(landmark_path, landmarks)

        # Save metadata
        metadata = [
            {
                'sample_id': s['sample_id'],
                'label': s['label'],
                'path': os.path.join(split_name, f"class{s['class_id']:03d}", f"{s['sample_id']}.npy")
            }
            for s in samples
        ]

        metadata_path = os.path.join(output_dir, f'{split_name}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  {split_name}: {len(samples)} samples saved to {split_dir}")

    # Save vocabulary
    vocab = {f'class{i:03d}': i for i in range(num_classes)}
    vocab_path = os.path.join(output_dir, 'vocabulary.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\n✓ Dataset created at {output_dir}")
    print(f"  Train: {len(split_samples['train'])} samples")
    print(f"  Val: {len(split_samples['val'])} samples")
    print(f"  Test: {len(split_samples['test'])} samples")
    print(f"  Vocabulary: {num_classes} classes")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic word-sign dataset")
    parser.add_argument('--num-classes', type=int, default=20,
                        help="Number of word classes")
    parser.add_argument('--samples-per-class', type=int, default=50,
                        help="Samples per class")
    parser.add_argument('--output-dir', default='data/synthetic_wordsigns',
                        help="Output directory")
    parser.add_argument('--input-dim', type=int, default=63,
                        help="Landmark dimension (default 63 for MediaPipe hands)")
    parser.add_argument('--base-seq-len', type=int, default=50,
                        help="Base sequence length before augmentation")
    args = parser.parse_args()

    create_dataset(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        output_dir=args.output_dir,
        input_dim=args.input_dim,
        base_seq_len=args.base_seq_len
    )


if __name__ == "__main__":
    main()
