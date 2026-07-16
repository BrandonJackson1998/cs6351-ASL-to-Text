"""Word-level sign recognition using temporal CNNs on landmark sequences.

Simple temporal model for WLASL word recognition:
- Input: T × 63 landmark sequence (MediaPipe hands)
- Architecture: 1D CNN + LSTM + classification head
- Output: Word class from vocabulary

This is a lightweight alternative to OpenHands ST-GCN that we can train ourselves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN_LSTM(nn.Module):
    """Temporal CNN + LSTM for word-sign recognition.

    Architecture:
    - 1D Conv layers to extract temporal features
    - Bi-LSTM to capture sequential dependencies
    - Classification head

    Args:
        num_classes: Number of word signs in vocabulary
        input_dim: Landmark dimension (default 63 for MediaPipe hands)
        hidden_dim: LSTM hidden size
        num_layers: Number of LSTM layers
    """

    def __init__(self, num_classes, input_dim=63, hidden_dim=256, num_layers=2):
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Temporal convolutions to extract features
        # Input: (batch, input_dim, T)
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)

        # Bi-LSTM for sequential modeling
        self.lstm = nn.LSTM(
            256, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, T, input_dim) landmark sequence

        Returns:
            logits: (batch, num_classes)
        """
        # Reshape for 1D conv: (batch, input_dim, T)
        x = x.transpose(1, 2)

        # Temporal convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Reshape back for LSTM: (batch, T', features)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Global average pooling over time
        # Average all timesteps to get fixed-size representation
        x = torch.mean(lstm_out, dim=1)  # (batch, hidden_dim*2)

        # Classification
        logits = self.fc(x)
        return logits


class SimpleTemporalCNN(nn.Module):
    """Simpler CNN-only model for faster training/inference.

    Good baseline before trying the full CNN+LSTM.
    """

    def __init__(self, num_classes, input_dim=63):
        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim

        # Temporal convolutions
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        x = x.transpose(1, 2)  # (batch, input_dim, T)
        x = self.conv_layers(x)
        logits = self.classifier(x)
        return logits


def create_model(model_type='cnn_lstm', num_classes=100, input_dim=63):
    """Factory function to create word-sign recognition models.

    Args:
        model_type: 'cnn_lstm' or 'cnn'
        num_classes: Vocabulary size (WLASL100, WLASL300, or WLASL2000)
        input_dim: Landmark dimension (63 for MediaPipe hands)

    Returns:
        PyTorch model
    """
    if model_type == 'cnn_lstm':
        return TemporalCNN_LSTM(num_classes, input_dim)
    elif model_type == 'cnn':
        return SimpleTemporalCNN(num_classes, input_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test model shapes
    batch_size = 4
    seq_len = 50  # 50 frames
    input_dim = 63
    num_classes = 100  # WLASL100

    x = torch.randn(batch_size, seq_len, input_dim)

    print("Testing SimpleTemporalCNN:")
    model_cnn = SimpleTemporalCNN(num_classes, input_dim)
    out = model_cnn(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

    print("\nTesting TemporalCNN_LSTM:")
    model_lstm = TemporalCNN_LSTM(num_classes, input_dim)
    out = model_lstm(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
