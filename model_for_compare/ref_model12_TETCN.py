# -*- coding: utf-8 -*-
"""
ref_model12_TETCN.py

TETCN reproduction (Sensors & Actuators B: Chemical 405 (2024) 135272)
- Transformer Encoder (TE) + Temporal Convolutional Network (TCN)
- Global Average Pooling + MLP head

Input: (B, C, T) = (batch, channel, token)  e.g. (B, 16, 400)

Paper-aligned hyperparams (Table 2):
- key dim = 9, num heads = 2  -> embed_dim = 18
- dropout = 0.49
- filter size (kernel) = 2
- dilation factors = [1, 2, 4]
- TCN filters = 64, matching the common TCN-layer default when the paper
  does not explicitly report nb_filters.
"""

from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Positional Encoding (sin/cos)
# -------------------------
class SinCosPositionalEncoding(nn.Module):
    """
    Sin/Cos positional encoding added to sequence.
    For TE input shape: (B, T, D)
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        pe = torch.zeros(self.max_len, self.d_model)  # (max_len, D)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model)
        )
        # pe[:, 0::2] = sin(pos * div_term)
        # pe[:, 1::2] = cos(pos * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        if x.dim() != 3:
            raise ValueError(f"[PosEnc] Expected (B,T,D), got {tuple(x.shape)}")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"[PosEnc] d_model mismatch: expected {self.d_model}, got {D}")
        if T > self.max_len:
            raise ValueError(f"[PosEnc] T={T} exceeds max_len={self.max_len}")
        return x + self.pe[:T].unsqueeze(0)  # (1,T,D)


# -------------------------
# Transformer Encoder Block (custom, simple & stable)
# -------------------------
class TEBlock(nn.Module):
    """
    One Transformer-Encoder block:
    - Multi-head self-attention
    - FFN (Dense->GeLU->Dense)
    - Residual + LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int = 16, dropout: float = 0.49):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


# -------------------------
# TCN building blocks (causal + dilation)
# -------------------------
class Chomp1d(nn.Module):
    """Remove padding on the right to keep causality."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp == 0:
            return x
        return x[:, :, :-self.chomp]


class TemporalBlock(nn.Module):
    """
    One TCN block:
    causal dilated conv -> LayerNorm -> ReLU -> Dropout
    (with residual)
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)

        # causal padding: pad_left = (k-1)*d, implemented by Conv1d padding then chomp right
        pad = (self.kernel_size - 1) * self.dilation

        self.conv1 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            dilation=self.dilation,
            bias=True,
        )
        self.chomp1 = Chomp1d(pad)
        self.conv2 = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=pad,
            dilation=self.dilation,
            bias=True,
        )
        self.chomp2 = Chomp1d(pad)

        # LayerNorm expects (B,T,C), so we transpose around it
        self.ln1 = nn.LayerNorm(self.channels)
        self.ln2 = nn.LayerNorm(self.channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        y = self.conv1(x)
        y = self.chomp1(y)  # (B,C,T) causal
        y = y.transpose(1, 2)      # (B,T,C)
        y = self.ln1(y)
        y = y.transpose(1, 2)      # (B,C,T)
        y = self.drop(self.act(y))

        y = self.conv2(y)
        y = self.chomp2(y)
        y = y.transpose(1, 2)
        y = self.ln2(y)
        y = y.transpose(1, 2)
        y = self.drop(self.act(y))
        return x + y  # residual


class TCN(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: List[int], dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            TemporalBlock(channels=channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# -------------------------
# TETCN main model
# -------------------------
class Ref12_TETCN(nn.Module):
    """
    Input: (B, C_in, T)
    TE uses embed_dim=18 to match key_dim=9 with num_heads=2 (paper Table 2).
    """
    def __init__(
        self,
        num_classes: int,
        C_in: int = 16,
        T: int = 400,
        te_embed_dim: int = 18,       # = 2*key_dim(9)
        te_heads: int = 2,
        te_blocks: int = 1,           # paper doesn't clearly state depth; 1 is safe baseline
        te_ffn_hidden: int = 16,      # "Dense(16)" style FFN
        dropout: float = 0.49,
        tcn_kernel: int = 2,          # filter size
        tcn_dilations: List[int] = (1, 2, 4),
        tcn_filters: int = 64,
        head_hidden: int = 64,        # your choice; head is not strictly specified
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.C_in = int(C_in)
        self.T = int(T)

        self.proj_in = nn.Linear(self.C_in, te_embed_dim)
        self.pos = SinCosPositionalEncoding(d_model=te_embed_dim, max_len=max(4096, self.T + 8))

        self.te = nn.Sequential(*[
            TEBlock(d_model=te_embed_dim, num_heads=te_heads, ffn_hidden=te_ffn_hidden, dropout=dropout)
            for _ in range(int(te_blocks))
        ])

        self.proj_tcn = nn.Linear(te_embed_dim, tcn_filters)

        self.tcn = TCN(channels=tcn_filters, kernel_size=tcn_kernel, dilations=list(tcn_dilations), dropout=dropout)

        # Average pooling over time + MLP head
        self.head = nn.Sequential(
            nn.Linear(tcn_filters, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) == (batch, channel, token)
        """
        if x.dim() != 3:
            raise ValueError(f"[Ref12_TETCN] Expected (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.C_in or T != self.T:
            raise ValueError(f"[Ref12_TETCN] Expected (C,T)=({self.C_in},{self.T}), got ({C},{T})")

        # (B,C,T) -> (B,T,C)
        xt = x.transpose(1, 2)

        # TE: project -> add pos -> transformer
        xt = self.proj_in(xt)         # (B,T,18)
        xt = self.pos(xt)             # (B,T,18)
        xt = self.te(xt)              # (B,T,18)
        xt = self.proj_tcn(xt)        # (B,T,tcn_filters)

        # back to (B,tcn_filters,T) for TCN
        y = xt.transpose(1, 2)
        y = self.tcn(y)

        # global average pooling over time
        feat = y.mean(dim=-1)

        logits = self.head(feat)      # (B,num_classes)
        return logits


def build_ref11(num_classes: int, C_in: int, T: int) -> nn.Module:
    return Ref12_TETCN(num_classes=num_classes, C_in=C_in, T=T)


def build_ref12(num_classes: int, C_in: int, T: int) -> nn.Module:
    return Ref12_TETCN(num_classes=num_classes, C_in=C_in, T=T)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ref12_TETCN(num_classes=8, C_in=16, T=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
