"""Test word-sign recognition pipeline with synthetic data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.word_sign_recognizer import SimpleTemporalCNN, TemporalCNN_LSTM, create_model


class SyntheticWLASLDataset(Dataset):
    """Synthetic dataset for testing."""

    def __init__(self, num_samples=100, num_classes=10, seq_len=50, input_dim=63):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Generate random landmarks and labels
        np.random.seed(42)
        self.landmarks = [
            np.random.randn(seq_len, input_dim).astype(np.float32)
            for _ in range(num_samples)
        ]
        self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.landmarks[idx]), self.labels[idx]


def test_model_forward():
    """Test that models can forward pass."""
    print("\n" + "="*60)
    print("Test 1: Model Forward Pass")
    print("="*60)

    batch_size = 8
    seq_len = 50
    input_dim = 63
    num_classes = 100

    x = torch.randn(batch_size, seq_len, input_dim)

    # Test SimpleTemporalCNN
    model_cnn = SimpleTemporalCNN(num_classes, input_dim)
    out_cnn = model_cnn(x)

    assert out_cnn.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out_cnn.shape}"
    print(f"✓ SimpleTemporalCNN: {x.shape} → {out_cnn.shape}")

    # Test TemporalCNN_LSTM
    model_lstm = TemporalCNN_LSTM(num_classes, input_dim)
    out_lstm = model_lstm(x)

    assert out_lstm.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out_lstm.shape}"
    print(f"✓ TemporalCNN_LSTM: {x.shape} → {out_lstm.shape}")

    print("✓ All models forward pass correctly\n")


def test_dataset_loading():
    """Test dataset and dataloader."""
    print("="*60)
    print("Test 2: Dataset Loading")
    print("="*60)

    dataset = SyntheticWLASLDataset(num_samples=100, num_classes=10, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Test one batch
    for landmarks, labels in dataloader:
        assert landmarks.shape == (16, 50, 63), f"Unexpected shape: {landmarks.shape}"
        assert labels.shape == (16,), f"Unexpected shape: {labels.shape}"
        print(f"✓ Batch: landmarks={landmarks.shape}, labels={labels.shape}")
        break

    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ DataLoader: {len(dataloader)} batches\n")


def test_training_step():
    """Test a single training step."""
    print("="*60)
    print("Test 3: Training Step")
    print("="*60)

    # Setup
    num_classes = 10
    model = SimpleTemporalCNN(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = SyntheticWLASLDataset(num_samples=32, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # One training step
    model.train()
    landmarks, labels = next(iter(dataloader))

    optimizer.zero_grad()
    logits = model(landmarks)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # Check predictions
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()

    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Accuracy (random): {acc:.2%}")
    print(f"✓ Training step successful\n")


def test_inference():
    """Test inference on single sample."""
    print("="*60)
    print("Test 4: Inference")
    print("="*60)

    num_classes = 10
    model = SimpleTemporalCNN(num_classes)
    model.eval()

    # Single sample
    landmarks = torch.randn(1, 50, 63)

    with torch.no_grad():
        logits = model(landmarks)
        probs = torch.softmax(logits, dim=1)
        top5_probs, top5_idx = torch.topk(probs, k=5, dim=1)

    print(f"✓ Input: {landmarks.shape}")
    print(f"✓ Logits: {logits.shape}")
    print(f"✓ Top-5 predictions:")
    for i in range(5):
        class_id = top5_idx[0, i].item()
        prob = top5_probs[0, i].item()
        print(f"    {i+1}. Class {class_id}: {prob:.3f}")

    print("✓ Inference successful\n")


def test_variable_length_sequences():
    """Test handling of variable-length sequences."""
    print("="*60)
    print("Test 5: Variable-Length Sequences")
    print("="*60)

    num_classes = 10
    model = SimpleTemporalCNN(num_classes)
    model.eval()

    # Test different sequence lengths
    seq_lengths = [30, 50, 100, 150]

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, 63)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, num_classes)
        print(f"✓ Sequence length {seq_len}: {x.shape} → {out.shape}")

    print("✓ Model handles variable-length sequences\n")


def test_model_factory():
    """Test model creation via factory function."""
    print("="*60)
    print("Test 6: Model Factory")
    print("="*60)

    # Test CNN model
    model_cnn = create_model('cnn', num_classes=100, input_dim=63)
    assert isinstance(model_cnn, SimpleTemporalCNN)
    print(f"✓ Created SimpleTemporalCNN: {sum(p.numel() for p in model_cnn.parameters()):,} params")

    # Test CNN+LSTM model
    model_lstm = create_model('cnn_lstm', num_classes=100, input_dim=63)
    assert isinstance(model_lstm, TemporalCNN_LSTM)
    print(f"✓ Created TemporalCNN_LSTM: {sum(p.numel() for p in model_lstm.parameters()):,} params")

    print("✓ Model factory working\n")


def test_top_k_accuracy():
    """Test top-K accuracy computation."""
    print("="*60)
    print("Test 7: Top-K Accuracy")
    print("="*60)

    # Synthetic predictions and labels
    logits = torch.randn(32, 10)  # 32 samples, 10 classes
    labels = torch.randint(0, 10, (32,))

    # Top-1 accuracy
    preds = torch.argmax(logits, dim=1)
    top1_acc = (preds == labels).float().mean().item()

    # Top-5 accuracy
    top5_preds = torch.topk(logits, k=5, dim=1).indices
    top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=1)
    top5_acc = top5_correct.float().mean().item()

    print(f"✓ Top-1 accuracy: {top1_acc:.2%}")
    print(f"✓ Top-5 accuracy: {top5_acc:.2%}")
    assert top5_acc >= top1_acc, "Top-5 should be >= Top-1"
    print("✓ Top-K accuracy computation correct\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Word-Sign Recognition Pipeline")
    print("="*60)

    test_model_forward()
    test_dataset_loading()
    test_training_step()
    test_inference()
    test_variable_length_sequences()
    test_model_factory()
    test_top_k_accuracy()

    print("="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nPipeline ready for real data.")
    print("Next: Download WLASL videos and extract landmarks.\n")
