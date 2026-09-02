# -*- coding: utf-8 -*-
"""SmellNet/ScentFormer adaptation for the local e-nose comparison pipeline.

Reference: https://github.com/MIT-MI/SmellNet

The repository describes ScentFormer as the temporal model used for sensor
time-series classification. Its public `models/models.py` implements the
sensor-only branch as a batch-first Transformer classifier over `(B, T, F)`.
This file adapts that model to the local project convention:

Input:  (B, 16, 400)
Output: (B, num_classes)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class Ref17_SmellNetScentFormer(nn.Module):
    """Sensor-only SmellNet Transformer adapted to `(B, C, T)` local tensors."""

    def __init__(
        self,
        num_channels: int = 16,
        num_classes: int = 8,
        seq_length: int = 400,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_positional_encoding: bool = True,
        use_cls_token: bool = False,
        pool: str = "mean",
        use_temporal_diff: bool = False,
    ):
        super().__init__()
        if pool not in ("mean", "cls"):
            raise ValueError("pool must be 'mean' or 'cls'")
        if use_cls_token is False and pool == "cls":
            raise ValueError("pool='cls' requires use_cls_token=True")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.num_channels = num_channels
        self.seq_length = seq_length
        self.use_cls_token = use_cls_token
        self.pool = pool
        self.use_temporal_diff = use_temporal_diff

        self.input_proj = nn.Sequential(
            nn.Linear(num_channels, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.pos = SinusoidalPositionalEncoding(model_dim, max_len=seq_length + 1) if use_positional_encoding else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    @staticmethod
    def _key_padding_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
        rng = torch.arange(time_steps, device=lengths.device).unsqueeze(0)
        return rng >= lengths.unsqueeze(1)

    def _maybe_temporal_diff(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_temporal_diff:
            return x
        first = torch.zeros_like(x[:, :1, :])
        return torch.cat([first, x[:, 1:, :] - x[:, :-1, :]], dim=1)

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,C,T), got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.seq_length:
            raise ValueError(
                f"Expected input (B,{self.num_channels},{self.seq_length}), got shape {tuple(x.shape)}"
            )

        x = x.permute(0, 2, 1).contiguous()  # (B, T, C)
        x = self._maybe_temporal_diff(x)
        x = self.input_proj(x)
        x = self.pos(x)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = self._key_padding_mask(lengths, x.size(1))

        if self.use_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if key_padding_mask is not None:
                pad0 = torch.zeros((key_padding_mask.size(0), 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([pad0, key_padding_mask], dim=1)

        h = self.transformer(x, src_key_padding_mask=key_padding_mask)
        if self.use_cls_token and self.pool == "cls":
            return h[:, 0]

        tokens = h[:, 1:] if self.use_cls_token else h
        if key_padding_mask is None:
            return tokens.mean(dim=1)

        mask = ~key_padding_mask
        if self.use_cls_token:
            mask = mask[:, 1:]
        denom = mask.float().sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tokens * mask.unsqueeze(-1).float()).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.dropout(self.forward_features(x, lengths))
        return self.classifier(feat)


if __name__ == "__main__":
    model = Ref17_SmellNetScentFormer()
    y = model(torch.randn(2, 16, 400))
    print(y.shape)
