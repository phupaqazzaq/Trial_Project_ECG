"""Patch-based Transformer (Milestone 4, main model).

The encoder block is written by hand (instead of nn.TransformerEncoderLayer)
so that we can return the attention weights for the explainability step.
"""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """One Transformer layer (pre-norm):
    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))
    """

    def __init__(self, d_model, n_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention: every token looks at every other token.
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + self.dropout(attn_out)          # residual connection 1
        x = x + self.dropout(self.mlp(self.norm2(x)))  # residual connection 2
        return x, attn_weights                  # attn_weights: (batch, heads, tokens, tokens)


class TransformerEncoder(nn.Module):
    """[CLS] token + positional embeddings + a stack of encoder blocks.

    Takes a sequence of token vectors (batch, n_tokens, d_model) and returns
    the [CLS] output plus the attention weights of every layer.
    Shared by the patch Transformer and the hybrid model.
    """

    def __init__(self, n_tokens, d_model=128, n_heads=4, n_layers=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))  # +1 for [CLS]
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, mlp_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        batch = tokens.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)   # one [CLS] per ECG
        x = torch.cat([cls, tokens], dim=1)          # (batch, n_tokens + 1, d_model)
        x = self.dropout(x + self.pos_embed)         # tell the model where each token is in time

        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            attentions.append(attn)

        cls_out = self.norm(x)[:, 0]                 # the [CLS] token summarizes the whole ECG
        return cls_out, attentions


class PatchTransformer(nn.Module):
    """Input (batch, 12, 1000) -> 5 logits.

    The 10 s signal is cut into non-overlapping patches of `patch_size` time steps
    (default 50 steps = 0.5 s -> 20 patches). Each patch contains all 12 leads and is
    flattened to patch_size * 12 values (600), then projected to d_model.
    """

    def __init__(self, n_leads=12, seq_len=1000, patch_size=50, n_classes=5,
                 d_model=128, n_heads=4, n_layers=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_leads, d_model)
        self.encoder = TransformerEncoder(self.n_patches, d_model, n_heads, n_layers, mlp_dim, dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_attention=False):
        batch = x.shape[0]
        x = x.transpose(1, 2)                         # (batch, 1000, 12): time first
        x = x.reshape(batch, self.n_patches, -1)      # (batch, 20, 600): cut into patches
        tokens = self.patch_embed(x)                  # (batch, 20, d_model)
        cls_out, attentions = self.encoder(tokens)
        logits = self.head(cls_out)                   # (batch, 5)
        if return_attention:
            return logits, attentions
        return logits
