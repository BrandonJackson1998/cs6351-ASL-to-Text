"""Bi-LSTM + CTC for fingerspelling — v2.

Compared to v1:
  - 3 layers (was 2)
  - 256 hidden (was 128)
  - LayerNorm on the input projection
  - Dropout 0.3 (was 0.2)

Output layout matches v1: log-probs in (T, B, V) for nn.CTCLoss.
"""

import torch
import torch.nn as nn

from src.preprocessing.combined_dataset import BLANK_IDX, NUM_CLASSES


class CTCLSTMv2(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.input_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        x = self.input_norm(x)
        x = torch.relu(self.input_proj(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        out = self.head_dropout(out)
        logits = self.head(out)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)  # (T, B, V) for CTCLoss
