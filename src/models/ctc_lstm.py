"""Bi-LSTM + CTC head for fingerspelling.

Input:  (B, T, 63)  per-frame MediaPipe landmarks (already wrist-normalized)
Output: (T, B, V)   log-probabilities over V = 27 (26 letters + blank)
                    in the time-first layout that nn.CTCLoss expects.
"""

import torch
import torch.nn as nn


# Letter vocabulary. Index 0 = blank (CTC requirement). 1..26 = A..Z.
BLANK_IDX = 0
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 1 + len(LETTERS)  # 27


def label_string_to_targets(s: str) -> torch.Tensor:
    return torch.tensor(
        [LETTERS.index(c) + 1 for c in s.upper() if c in LETTERS],
        dtype=torch.long,
    )


def targets_to_string(targets) -> str:
    return "".join(LETTERS[t - 1] for t in targets if t != BLANK_IDX and 1 <= t <= len(LETTERS))


class CTCLSTM(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_layers=2, dropout=0.2,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        logits = self.head(out)  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)  # → (T, B, V) for nn.CTCLoss
