"""1D-CNN baseline (Milestone 3)."""

import torch.nn as nn


def conv_block(in_channels, out_channels):
    """Conv1d(kernel 7) -> BatchNorm -> ReLU -> MaxPool(2). Halves the time length."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class CNN1D(nn.Module):
    """Input (batch, 12, 1000) -> 5 logits.

    Time length through the blocks: 1000 -> 500 -> 250 -> 125 -> 62.
    Channels: 12 -> 32 -> 64 -> 128 -> 128.
    """

    def __init__(self, n_leads=12, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(n_leads, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average over time
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)          # (batch, 128, 62)
        x = self.pool(x).squeeze(-1)  # (batch, 128)
        return self.head(self.dropout(x))  # (batch, 5) logits, no sigmoid here
