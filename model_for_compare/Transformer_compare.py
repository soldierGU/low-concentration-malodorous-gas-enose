# -*- coding: utf-8 -*-
"""
TransformerEncoder.py

Transformer Encoder classifier baseline for E-nose time series.
Input : (B, C, T) = (B, 16, 400)
Output: (B, num_classes) logits

Pipeline:
  - Tokenize by 1D conv patch embedding (kernel=P, stride=S) -> tokens N
  - Add sinusoidal positional encoding
  - TransformerEncoder (PyTorch)
  - Global average pooling over tokens
  - MLP head
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinCosPositionalEncoding(nn.Module):
    """Sin/Cos positional encoding for (B, N, D)."""
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        if x.dim() != 3:
            raise ValueError(f"[PosEnc] Expected (B,N,D), got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.d_model:
            raise ValueError(f"[PosEnc] d_model mismatch: expected {self.d_model}, got {D}")
        if N > self.max_len:
            raise ValueError(f"[PosEnc] N={N} exceeds max_len={self.max_len}")
        return x + self.pe[:N].unsqueeze(0)


class PatchEmbedding1D(nn.Module):
    """
    Convert (B, C, T) to tokens (B, N, D) using Conv1d.
    """
    def __init__(self, C_in: int, d_model: int, patch_size: int = 8, stride: int = 4):
        super().__init__()
        self.C_in = int(C_in)
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.stride = int(stride)

        self.proj = nn.Conv1d(self.C_in, self.d_model,
                              kernel_size=self.patch_size, stride=self.stride,
                              padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        if x.dim() != 3:
            raise ValueError(f"[PatchEmbed] Expected (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.C_in:
            raise ValueError(f"[PatchEmbed] Expected C={self.C_in}, got {C}")

        y = self.proj(x)          # (B, D, N)
        y = y.transpose(1, 2)     # (B, N, D)
        return y


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        C_in: int = 16,
        T: int = 400,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_size: int = 8,
        stride: int = 4,
        pooling: str = "mean",   # "mean" or "cls"
        head_hidden: int = 128,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.C_in = int(C_in)
        self.T = int(T)
        self.d_model = int(d_model)
        self.pooling = str(pooling)

        # Tokenize
        self.patch = PatchEmbedding1D(C_in=self.C_in, d_model=self.d_model,
                                      patch_size=patch_size, stride=stride)

        # Estimate max token length for PE buffer (safe upper bound)
        # N = floor((T - P)/S) + 1
        N = (self.T - patch_size) // stride + 1
        self.pos = SinCosPositionalEncoding(d_model=self.d_model, max_len=max(2048, N + 8))

        # Optional CLS token
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 16, 400)
        """
        if x.dim() != 3:
            raise ValueError(f"[Ref15] Expected (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.C_in or T != self.T:
            raise ValueError(f"[Ref15] Expected (C,T)=({self.C_in},{self.T}), got ({C},{T})")

        tok = self.patch(x)          # (B, N, D)
        tok = self.pos(tok)

        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)     # (B,1,D)
            tok = torch.cat([cls, tok], dim=1)         # (B,1+N,D)

        z = self.encoder(tok)        # (B, N', D)

        if self.cls_token is not None:
            feat = z[:, 0, :]        # CLS pooling
        else:
            feat = z.mean(dim=1)     # mean pooling over tokens

        logits = self.head(feat)
        return logits


def build_ref15(num_classes: int, C_in: int, T: int) -> nn.Module:
    return TransformerEncoder(num_classes=num_classes, C_in=C_in, T=T)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerEncoder(num_classes=8, C_in=16, T=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
