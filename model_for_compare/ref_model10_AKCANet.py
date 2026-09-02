# -*- coding: utf-8 -*-
"""
ref_model10_AKCANet.py

Reproduction of AKCA-Net (Food Chemistry 2024) for E-nose classification.
Input: (B, C, T)  == (batch, channel/sensor, token/time)

Backbone:
  (B,C,T) -> (B,1,T,C)
  1x1 pointwise conv -> 3x3 conv -> AKCA attention -> 3x3 maxpool -> GAP -> FC

AKCA:
  GAP/GMP -> Conv1d(k=K) -> interleave to 2C -> downsample to C (stride=2) -> Conv1d(k=K) -> sigmoid -> reweight

No padding="same" is used (works with older PyTorch versions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AKCA(nn.Module):
    """
    Adaptive Kernel Channel Attention
    Input : (B, C, H, W)
    Output: (B, C, H, W)
    """
    def __init__(self, channels: int, k: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.k = int(k)

        # conv over channel statistics (length = C)
        # padding = (k//2) keeps length approximately; for even k, length may shift by 1, we will crop/pad.
        self.conv_stat = nn.Conv1d(1, 1, kernel_size=self.k, stride=1, padding=self.k // 2, bias=True)

        # downsample 2C -> C (stride=2) without "same"
        # use k=1 so output length = floor((2C-1)/2)+1 = C exactly
        self.down = nn.Conv1d(1, 1, kernel_size=1, stride=2, padding=0, bias=True)

        # refine on length=C
        self.refine = nn.Conv1d(1, 1, kernel_size=self.k, stride=1, padding=self.k // 2, bias=True)

    @staticmethod
    def _fix_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Ensure last dimension equals target_len by cropping or right-padding with zeros.
        x: (B, 1, L)
        """
        L = x.size(-1)
        if L == target_len:
            return x
        if L > target_len:
            return x[..., :target_len]
        # pad right
        pad = target_len - L
        return F.pad(x, (0, pad), mode="constant", value=0.0)

    @staticmethod
    def _interleave(a_max: torch.Tensor, a_avg: torch.Tensor) -> torch.Tensor:
        """
        a_max/a_avg: (B, C)
        -> (B, 2C) interleaved: (max1, avg1, ..., maxC, avgC)
        """
        B, C = a_max.shape
        return torch.stack([a_max, a_avg], dim=-1).reshape(B, 2 * C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"[AKCA] Expected (B,C,H,W), got {tuple(x.shape)}")
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"[AKCA] Expected channels={self.channels}, got {C}")

        # GAP / GMP -> (B,C)
        z_avg = x.mean(dim=(2, 3))
        z_max = x.amax(dim=(2, 3))

        # Conv1d over length=C
        a_avg = self.conv_stat(z_avg.unsqueeze(1))   # (B,1,L1)
        a_max = self.conv_stat(z_max.unsqueeze(1))   # (B,1,L1)
        a_avg = self._fix_length(a_avg, C).squeeze(1)  # (B,C)
        a_max = self._fix_length(a_max, C).squeeze(1)  # (B,C)

        # interleave -> (B, 2C)
        a_fusion = self._interleave(a_max, a_avg)      # (B,2C)

        # downsample 2C -> C exactly
        a_w = self.down(a_fusion.unsqueeze(1))          # (B,1,C)

        # refine (keep length ~C, then fix)
        a_w = self.refine(a_w)                          # (B,1,L2)
        a_w = self._fix_length(a_w, C)                  # (B,1,C)

        # sigmoid weights and reweight
        w = torch.sigmoid(a_w).view(B, C, 1, 1)
        return x * w


class Ref10_AKCA_Net(nn.Module):
    """
    Input: (B, C_in, T)
    """
    def __init__(self, num_classes: int, C_in: int, T: int, embed_dim: int = 32, akca_k: int = 4):
        super().__init__()
        self.num_classes = int(num_classes)
        self.C_in = int(C_in)
        self.T = int(T)
        self.embed_dim = int(embed_dim)

        # (B,1,T,C) -> embed_dim
        self.point = nn.Conv2d(1, self.embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.akca = AKCA(channels=self.embed_dim, k=akca_k)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"[Ref10_AKCA_Net] Expected (B,C,T), got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.C_in or T != self.T:
            raise ValueError(f"[Ref10_AKCA_Net] Expected (C,T)=({self.C_in},{self.T}), got ({C},{T})")

        # -> (B,1,T,C)
        x2d = x.permute(0, 2, 1).unsqueeze(1)

        y = F.relu(self.point(x2d))
        y = F.relu(self.conv3(y))
        y = self.akca(y)
        y = self.pool(y)

        y = self.gap(y).flatten(1)
        logits = self.fc(y)
        return logits


def build_ref10(num_classes: int, C_in: int, T: int) -> nn.Module:
    return Ref10_AKCA_Net(num_classes=num_classes, C_in=C_in, T=T, embed_dim=32, akca_k=4)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ref10_AKCA_Net(num_classes=8, C_in=16, T=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
