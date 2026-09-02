# -*- coding: utf-8 -*-
"""
ref_model9_teaCNN.py

2D-CNN baseline (paper-style) for MS-TS-DDA comparison.
Input: (B, C, T)  i.e., (batch, channel, token_length)

Internal:
  (B, C, T) -> (B, 1, T, C)
  Conv2d(kernel=(kernel_t, 1), stride=(stride_t, 1))  # conv over T only
  ReLU -> Flatten -> FC -> ReLU -> FC -> logits
"""
# Description of tea quality using deep learning and multi-sensor
# feature fusion
import torch
import torch.nn as nn
import torch.nn.functional as F


class Ref9_TeaCNN2D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        C: int,
        T: int,
        kernel_t: int = 215,
        stride_t: int = 20,
        conv_out_channels: int = 1,
        hidden: int = 64,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.C = int(C)
        self.T = int(T)
        self.kernel_t = int(kernel_t)
        self.stride_t = int(stride_t)
        self.conv_out_channels = int(conv_out_channels)
        self.hidden = int(hidden)

        # Conv only along T (height), keep width=C unchanged
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.conv_out_channels,
            kernel_size=(self.kernel_t, 1),
            stride=(self.stride_t, 1),
            padding=(0, 0),
            bias=True,
        )

        # Compute conv output size deterministically
        # H_out = floor((T - kernel_t)/stride_t) + 1
        H_out = (self.T - self.kernel_t) // self.stride_t + 1
        if H_out <= 0:
            raise ValueError(
                f"[Ref9_TeaCNN2D] Invalid conv output: T={self.T}, kernel_t={self.kernel_t}, stride_t={self.stride_t}"
            )

        feat_dim = self.conv_out_channels * H_out * self.C

        self.fc1 = nn.Linear(feat_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        if x.dim() != 3:
            raise ValueError(f"[Ref9_TeaCNN2D] Expected (B,C,T), got {tuple(x.shape)}")

        B, C, T = x.shape
        if C != self.C or T != self.T:
            raise ValueError(f"[Ref9_TeaCNN2D] Expected (C,T)=({self.C},{self.T}), got ({C},{T})")

        # -> (B,1,T,C)
        x2d = x.permute(0, 2, 1).unsqueeze(1)

        y = F.relu(self.conv(x2d))
        y = y.flatten(1)

        y = F.relu(self.fc1(y))
        logits = self.fc2(y)
        return logits


def build_ref9(num_classes: int, C: int, T: int,
               kernel_t: int = 215, stride_t: int = 20) -> nn.Module:
    """
    Helper for your compare framework.
    """
    return Ref9_TeaCNN2D(
        num_classes=num_classes,
        C=C, T=T,
        kernel_t=kernel_t,
        stride_t=stride_t,
        conv_out_channels=1,
        hidden=64,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ref9_TeaCNN2D(num_classes=8, C=16, T=400, kernel_t=215, stride_t=20).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
