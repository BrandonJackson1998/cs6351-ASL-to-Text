"""
LSTM Model (Phase 2)
Sequence model for dynamic word-level sign recognition.
Input: Temporal sequence of holistic landmarks per frame.
Output: Word-level sign class.
"""

import torch
import torch.nn as nn


class SignLanguageLSTM(nn.Module):
    """LSTM for classifying ASL word signs from temporal landmark sequences."""

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=100, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, input_dim)
        output, (h_n, _) = self.lstm(x)
        # Use final hidden states from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        return self.classifier(h_cat)
