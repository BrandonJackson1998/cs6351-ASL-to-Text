"""Train word-sign recognition model.

Usage:
    python -m src.models.train_word_sign \
        --data-dir data/synthetic_wordsigns \
        --model-type cnn \
        --experiment synthetic_wlasl20 \
        --epochs 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.word_sign_recognizer import create_model


class WordSignDataset(Dataset):
    """Dataset for word-sign recognition."""

    def __init__(self, data_dir, split='train', max_len=None):
        self.data_dir = data_dir
        self.split = split

        # Load metadata
        metadata_path = os.path.join(data_dir, f'{split}.json')
        with open(metadata_path) as f:
            self.samples = json.load(f)

        # Load vocabulary
        vocab_path = os.path.join(data_dir, 'vocabulary.json')
        with open(vocab_path) as f:
            self.vocabulary = json.load(f)

        self.num_classes = len(self.vocabulary)
        self.max_len = max_len

        print(f"Loaded {split} set: {len(self.samples)} samples, {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']

        # Load landmarks
        landmark_path = os.path.join(self.data_dir, sample['path'])
        landmarks = np.load(landmark_path).astype(np.float32)

        # Pad or truncate to max_len if specified
        if self.max_len:
            if len(landmarks) < self.max_len:
                # Pad with last frame
                pad_len = self.max_len - len(landmarks)
                padding = np.repeat(landmarks[-1:], pad_len, axis=0)
                landmarks = np.vstack([landmarks, padding])
            elif len(landmarks) > self.max_len:
                # Truncate
                landmarks = landmarks[:self.max_len]

        return torch.from_numpy(landmarks), label


def collate_variable_length(batch):
    """Collate function for variable-length sequences.

    Pads sequences to the longest in the batch.
    """
    landmarks_list, labels = zip(*batch)

    # Find max length in batch
    max_len = max(lm.shape[0] for lm in landmarks_list)

    # Pad all sequences
    padded = []
    for lm in landmarks_list:
        if lm.shape[0] < max_len:
            pad_len = max_len - lm.shape[0]
            padding = lm[-1:].repeat(pad_len, 1)
            lm = torch.cat([lm, padding], dim=0)
        padded.append(lm)

    landmarks = torch.stack(padded)
    labels = torch.tensor(labels, dtype=torch.long)

    return landmarks, labels


def compute_top_k_accuracy(logits, labels, k=5):
    """Compute top-K accuracy."""
    _, top_k_preds = torch.topk(logits, k=k, dim=1)
    correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for landmarks, labels in pbar:
        landmarks, labels = landmarks.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(landmarks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.3f}'})

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0

    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for landmarks, labels in dataloader:
            landmarks, labels = landmarks.to(device), labels.to(device)

            logits = model(landmarks)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total += labels.size(0)

            # Top-1 accuracy
            preds = torch.argmax(logits, dim=1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5 accuracy
            correct_top5 += compute_top_k_accuracy(logits, labels, k=min(5, logits.size(1))) * labels.size(0)

    avg_loss = total_loss / len(dataloader)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    return avg_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Train word-sign recognition model")
    parser.add_argument('--data-dir', required=True, help="Dataset directory")
    parser.add_argument('--model-type', choices=['cnn', 'cnn_lstm'], default='cnn')
    parser.add_argument('--experiment', required=True, help="Experiment name")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-len', type=int, default=None,
                        help="Max sequence length (None = variable length)")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    exp_dir = os.path.join('experiments', args.experiment)
    os.makedirs(exp_dir, exist_ok=True)

    print("="*60)
    print(f"Training Word-Sign Recognition: {args.experiment}")
    print("="*60)
    print(f"Model: {args.model_type}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*60)

    # Load datasets
    train_dataset = WordSignDataset(args.data_dir, 'train', max_len=args.max_len)
    val_dataset = WordSignDataset(args.data_dir, 'val', max_len=args.max_len)
    test_dataset = WordSignDataset(args.data_dir, 'test', max_len=args.max_len)

    num_classes = train_dataset.num_classes

    # DataLoaders
    collate_fn = None if args.max_len else collate_variable_length
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Model
    model = create_model(args.model_type, num_classes=num_classes, input_dim=63)
    model = model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                             patience=5)

    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5': []}

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5'].append(val_top5)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc (top-1): {val_acc:.4f}, Val Acc (top-5): {val_top5:.4f}")

        # Scheduler step
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_top5': val_top5,
            }, os.path.join(exp_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_acc={val_acc:.4f})")

        # Also save latest model (in case no validation set)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
        }, os.path.join(exp_dir, 'latest_model.pth'))

    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)

    # Load best model if available, otherwise use latest
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    latest_model_path = os.path.join(exp_dir, 'latest_model.pth')

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        print("Loaded best_model.pth")
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path)
        print("Loaded latest_model.pth (no validation set)")
    else:
        print("No checkpoint found, using current model state")
        checkpoint = None

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_top5 = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc (top-1): {test_acc:.4f}")
    print(f"Test Acc (top-5): {test_top5:.4f}")

    # Save results
    results = {
        'experiment': args.experiment,
        'model_type': args.model_type,
        'num_classes': num_classes,
        'epochs': args.epochs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_top5': test_top5,
        'history': history
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Training complete. Results saved to {exp_dir}")


if __name__ == "__main__":
    main()
