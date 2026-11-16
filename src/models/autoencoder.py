"""Milestone 8: 1D convolutional autoencoder for anomaly detection.

Idea: the model must squeeze an ECG through a narrow "bottleneck" and rebuild it.
It is trained on NORMAL ECGs only, so it learns to rebuild normal heartbeats well.
An ECG it rebuilds badly (high reconstruction error) is flagged as unusual.
"""

import torch.nn as nn


def down_block(in_channels, out_channels):
    """Conv with stride 2: halves the number of time steps."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


def up_block(in_channels, out_channels):
    """Transposed conv with stride 2: doubles the number of time steps."""
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


class ConvAutoencoder(nn.Module):
    """Input and output: (batch, 12, 1000).

    Encoder: 1000 -> 500 -> 250 -> 125 time steps, ending with `bottleneck_channels`.
    With 8 channels the bottleneck holds 125 x 8 = 1,000 numbers for an ECG of
    12,000 numbers (12x compression), so the model cannot simply copy the input.
    Decoder: the mirror image, 125 -> 250 -> 500 -> 1000.
    """

    def __init__(self, bottleneck_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            down_block(12, 32),
            down_block(32, 64),
            # last encoder layer: plain conv, no ReLU, so the code can be any real number
            nn.Conv1d(64, bottleneck_channels, kernel_size=7, stride=2, padding=3),
        )
        self.decoder = nn.Sequential(
            up_block(bottleneck_channels, 64),
            up_block(64, 32),
            # last decoder layer: no activation, because standardized ECG values
            # can be positive or negative
            nn.ConvTranspose1d(32, 12, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
