"""
MLP Baseline Model (Phase 1)
Dense neural network for static fingerspelling classification.
Input: 63 features (21 hand keypoints x 3 coordinates)
Output: 29 classes (A-Z + SPACE, DELETE, NOTHING)
"""

import torch.nn as nn


class FingerspellingMLP(nn.Module):
    """MLP for classifying static ASL alphabet signs from hand landmarks."""

    def __init__(self, input_dim=63, num_classes=29, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
