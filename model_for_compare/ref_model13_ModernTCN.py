# -*- coding: utf-8 -*-
"""
ref_model13_ModernTCN.py

ModernTCN classifier adapted for E-nose data.
Input: (B, C, T) = (batch, channel/variables M, token/time length L)
Example: (B, 16, 400)

Pipeline (paper-style):
  Patchify variable-independent embedding -> ModernTCN backbone -> Flatten -> Projection -> logits
(Softmax is applied implicitly by CrossEntropyLoss during training.)
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_to_patchify(L: int, P: int, S: int) -> int:
    """Right-pad so that (L_padded - P) % S == 0."""
    if L < P:
        return P - L
    rem = (L - P) % S
    return (S - rem) % S


class PatchifyVarIndEmbed(nn.Module):
    """
    Variable-independent patch embedding via Conv1d:
      x: (B, M, L) -> (B, M, D, N)
    """
    def __init__(self, M: int, D: int, P: int, S: int):
        super().__init__()
        self.M, self.D, self.P, self.S = int(M), int(D), int(P), int(S)
        # shared weights across variables (equivalent to apply same conv to each variable separately)
        self.stem = nn.Conv1d(1, self.D, kernel_size=self.P, stride=self.S, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"[Embed] Expected (B,M,L), got {tuple(x.shape)}")
        B, M, L = x.shape
        if M != self.M:
            raise ValueError(f"[Embed] Expected M={self.M}, got {M}")

        pad_r = _pad_to_patchify(L, self.P, self.S)
        if pad_r > 0:
            x = F.pad(x, (0, pad_r), mode="constant", value=0.0)

        x = x.reshape(B * M, 1, x.size(-1))     # (B*M,1,Lp)
        y = self.stem(x)                        # (B*M,D,N)
        N = y.size(-1)
        return y.reshape(B, M, self.D, N)       # (B,M,D,N)


class LayerNormChannel(nn.Module):
    """LayerNorm over channel dimension for x: (B,C,T)."""
    def __init__(self, C: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(C, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ModernTCNBlock(nn.Module):
    """
    One ModernTCN block operating on z: (B,M,D,N).
    Internally merge (M,D) -> channels C=M*D, conv along N.
    """
    def __init__(self, M: int, D: int,
                 large_kernel: int = 51,
                 ffn_ratio: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.M, self.D = int(M), int(D)
        self.C = self.M * self.D
        self.large_kernel = int(large_kernel)
        self.ffn_ratio = int(ffn_ratio)
        self.dropout = float(dropout)

        # DWConv: groups=C, temporal mixing only
        pad = self.large_kernel // 2
        self.dwconv = nn.Conv1d(self.C, self.C, kernel_size=self.large_kernel, stride=1,
                                padding=pad, groups=self.C, bias=True)
        self.norm1 = LayerNormChannel(self.C)
        self.act = nn.GELU()
        self.drop = nn.Dropout(self.dropout)

        # ConvFFN1: groups=M (per-variable feature mixing)
        hidden = self.ffn_ratio * self.C
        self.pw1_1 = nn.Conv1d(self.C, hidden, kernel_size=1, groups=self.M, bias=True)
        self.pw1_2 = nn.Conv1d(hidden, self.C, kernel_size=1, groups=self.M, bias=True)
        self.norm2 = LayerNormChannel(self.C)

        # ConvFFN2: groups=D (per-feature cross-variable mixing) after (M,D)<->(D,M) reindex
        self.pw2_1 = nn.Conv1d(self.C, hidden, kernel_size=1, groups=self.D, bias=True)
        self.pw2_2 = nn.Conv1d(hidden, self.C, kernel_size=1, groups=self.D, bias=True)
        self.norm3 = LayerNormChannel(self.C)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4:
            raise ValueError(f"[Block] Expected (B,M,D,N), got {tuple(z.shape)}")
        B, M, D, N = z.shape
        if M != self.M or D != self.D:
            raise ValueError(f"[Block] Expected (M,D)=({self.M},{self.D}), got ({M},{D})")

        x = z.reshape(B, self.C, N)

        # DWConv + residual
        y = self.dwconv(x)
        if y.size(-1) != N:
            y = y[..., :N]
        y = self.drop(self.act(self.norm1(y)))
        x = x + y

        # ConvFFN1 + residual
        y = self.pw1_2(self.drop(self.act(self.pw1_1(x))))
        y = self.drop(self.act(self.norm2(y)))
        x = x + y

        # ConvFFN2 with channel reindex (M,D)->(D,M)
        x_dm = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, self.C, N)
        y = self.pw2_2(self.drop(self.act(self.pw2_1(x_dm))))
        y = self.drop(self.act(self.norm3(y)))
        x_dm = x_dm + y
        out = x_dm.reshape(B, D, M, N).permute(0, 2, 1, 3)  # (B,M,D,N)

        return out


class Ref13_ModernTCN_Classifier(nn.Module):
    """
    Input:  x (B,M,L) == (B,16,400)
    Output: logits (B,num_classes)
    """
    def __init__(self,
                 num_classes: int,
                 M: int = 16,
                 L: int = 400,
                 D: int = 64,
                 P: int = 8,
                 S: int = 4,
                 blocks: int = 2,
                 large_kernel: int = 51,
                 ffn_ratio: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.M, self.L = int(M), int(L)
        self.D, self.P, self.S = int(D), int(P), int(S)

        self.embed = PatchifyVarIndEmbed(M=self.M, D=self.D, P=self.P, S=self.S)

        # infer N for fixed (L,P,S)
        pad_r = _pad_to_patchify(self.L, self.P, self.S)
        Lp = self.L + pad_r
        self.N = (Lp - self.P) // self.S + 1

        self.backbone = nn.ModuleList([
            ModernTCNBlock(M=self.M, D=self.D,
                           large_kernel=large_kernel,
                           ffn_ratio=ffn_ratio,
                           dropout=dropout)
            for _ in range(int(blocks))
        ])

        # Paper-style head: flatten (M*D*N) -> linear projection -> logits
        self.proj = nn.Linear(self.M * self.D * self.N, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"[ModernTCN] Expected (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.M or T != self.L:
            raise ValueError(f"[ModernTCN] Expected (C,T)=({self.M},{self.L}), got ({C},{T})")

        z = self.embed(x)  # (B,M,D,N)
        for blk in self.backbone:
            z = blk(z)

        feat = z.reshape(B, -1)       # (B, M*D*N)
        logits = self.proj(feat)      # (B, num_classes)
        return logits


def build_ref13(num_classes: int, C_in: int, T: int) -> nn.Module:
    # helper for your compare framework
    assert C_in == 16 and T == 400, "Adjust M/L if your dataset changes."
    return Ref13_ModernTCN_Classifier(num_classes=num_classes, M=C_in, L=T)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ref13_ModernTCN_Classifier(num_classes=8, M=16, L=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
