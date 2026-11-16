"""Hybrid CNN + Transformer (Milestone 5).

A small CNN first turns the raw signal into local features (it is good at
finding shapes like the QRS spike). Each remaining time step becomes one token,
and the same Transformer encoder then relates the tokens across the 10 seconds.
"""

import torch.nn as nn

from src.models.cnn import conv_block
from src.models.transformer import TransformerEncoder


class HybridCNNTransformer(nn.Module):
    """Input (batch, 12, 1000) -> 5 logits.

    CNN front-end: 3 conv blocks, time 1000 -> 500 -> 250 -> 125.
    So the Transformer sees 125 tokens, each covering 0.08 s of signal.
    """

    def __init__(self, n_leads=12, seq_len=1000, n_classes=5,
                 d_model=128, n_heads=4, n_layers=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            conv_block(n_leads, 32),
            conv_block(32, 64),
            conv_block(64, d_model),
        )
        self.n_tokens = seq_len // 8  # three MaxPool(2) layers -> 125 tokens
        self.encoder = TransformerEncoder(self.n_tokens, d_model, n_heads, n_layers, mlp_dim, dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_attention=False):
        features = self.cnn(x)                 # (batch, d_model, 125)
        tokens = features.transpose(1, 2)      # (batch, 125, d_model): one token per time step
        cls_out, attentions = self.encoder(tokens)
        logits = self.head(cls_out)
        if return_attention:
            return logits, attentions
        return logits
