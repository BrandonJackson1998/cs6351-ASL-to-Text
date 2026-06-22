"""Train EfficientNet-B0 on hand crop images for ASL letter classification.

Inspired by Kaggle 99.96% F1 approach:
- EfficientNet-B0 pretrained on ImageNet
- Custom classification head with dropout
- Aggressive augmentation (rotation, shift, zoom, flip)
- Adamax optimizer with early stopping

Expected: 94-96% test accuracy (vs 83% with RF on landmarks)
"""

import argparse
import json
import os
import time
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class HandCropDataset(Dataset):
    """Dataset for hand crop images."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 with custom classification head.

    Architecture matches Kaggle approach:
    - Pretrained EfficientNet-B0 backbone
    - BatchNorm + Linear(256) + ReLU + Dropout(0.4) + Linear(26)
    """

    def __init__(self, num_classes=26, pretrained=True, dropout=0.4):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='max'  # Max pooling like Kaggle
        )

        # Custom head
        in_features = self.backbone.num_features  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features, momentum=0.99, eps=0.001),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_transforms(train=True, img_size=200):
    """Get augmentation transforms.

    Training augmentations (from Kaggle):
    - Horizontal flip
    - Rotation ±20°
    - Shift ±20%
    - Zoom ±20%

    Test: Only normalization
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=0,
                p=0.7
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])


def load_dataset(data_dir):
    """Load all hand crop images and create train/val/test splits."""
    data_dir = Path(data_dir)

    image_paths = []
    labels = []
    letter_to_idx = {}

    # Collect all images
    letters = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and len(d.name) == 1])

    for idx, letter in enumerate(letters):
        letter_to_idx[letter] = idx
        letter_dir = data_dir / letter

        for img_path in letter_dir.glob('*.jpg'):
            image_paths.append(str(img_path))
            labels.append(idx)

    print(f"Loaded {len(image_paths)} images from {len(letters)} classes")

    # Split: 60% train, 20% val, 20% test (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.4, stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), letter_to_idx


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Train')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet on hand crops')
    parser.add_argument('--data-dir', type=str, default='data/hand_crops_kaggle',
                        help='Directory with hand crop images')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (default: efficientnet_YYYYMMDD_HHMMSS)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs (default: 40)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate (default: 0.005)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate (default: 0.4)')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience (default: 8)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda, or mps')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load data
    print("\nLoading dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), letter_to_idx = load_dataset(args.data_dir)

    # Create datasets
    train_dataset = HandCropDataset(X_train, y_train, transform=get_transforms(train=True))
    val_dataset = HandCropDataset(X_val, y_val, transform=get_transforms(train=False))
    test_dataset = HandCropDataset(X_test, y_test, transform=get_transforms(train=False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    print("\nCreating model...")
    model = EfficientNetClassifier(
        num_classes=len(letter_to_idx),
        pretrained=True,
        dropout=args.dropout
    ).to(device)

    print(f"Model: EfficientNet-B0 with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    # Create experiment directory
    if args.exp_name:
        exp_dir = Path('experiments') / args.exp_name
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        exp_dir = Path('experiments') / f'efficientnet_{timestamp}'

    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment directory: {exp_dir}")

    # Save config
    config = {
        'model': 'EfficientNet-B0',
        'data_dir': args.data_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'dropout': args.dropout,
        'patience': args.patience,
        'num_classes': len(letter_to_idx),
        'letter_to_idx': letter_to_idx,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
    }

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, exp_dir / 'best_model.pth')

            print(f"✓ New best model! Val Acc: {100*val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print("\nEarly stopping!")
                break

    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(exp_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test
    print("Evaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nFinal Results:")
    print(f"Best Val Acc: {100*best_val_acc:.2f}%")
    print(f"Test Acc: {100*test_acc:.2f}%")

    # Save metrics
    metrics = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'num_epochs': len(history),
        'history': history
    }

    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to: {exp_dir}")
    print(f"  - best_model.pth")
    print(f"  - config.json")
    print(f"  - metrics.json")


if __name__ == '__main__':
    main()
