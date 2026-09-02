# -*- coding: utf-8 -*-
"""TimeMIL adaptation for the local e-nose comparison pipeline.

Reference: https://github.com/xiwenc1/TimeMIL
Paper: TimeMIL: Advancing Multivariate Time Series Classification via a
Time-aware Multiple Instance Learning (ICML 2024),
https://arxiv.org/abs/2405.03140

This file adapts the published TimeMIL architecture and its public reference
implementation to the input/output conventions used by this project. The
upstream repository did not include an explicit software license when checked
on 2026-09-02. This file is included for transparent academic comparison and
reproducibility; it does not imply endorsement by the TimeMIL authors or grant
rights to their original materials. Consult the upstream authors and repository
before reusing or redistributing code derived from the upstream implementation.
See ``TIMEMIL_NOTICE.md`` for details.

Input:  (B, 16, 400)
Output: (B, num_classes)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 16, bottleneck_channels: int = 16):
        super().__init__()
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding="same")
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=10, padding="same"),
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=20, padding="same"),
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=40, padding="same"),
            ]
        )
        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same"),
        )
        self.activation = nn.Sequential(nn.BatchNorm1d(4 * out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        z = torch.cat(
            [
                self.conv_layers[0](x_bottleneck),
                self.conv_layers[1](x_bottleneck),
                self.conv_layers[2](x_bottleneck),
                self.max_pooling_w_bottleneck(x),
            ],
            dim=1,
        )
        return self.activation(z)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 16, bottleneck_channels: int = 16, n_modules: int = 3):
        super().__init__()
        modules = []
        for i in range(n_modules):
            modules.append(
                InceptionModule(
                    in_channels=in_channels if i == 0 else out_channels * 4,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                )
            )
        self.inception_modules = nn.Sequential(*modules)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, 4 * out_channels, kernel_size=1, padding="same"),
            nn.BatchNorm1d(4 * out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.inception_modules(x) + self.residual(x))


class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(self, n_in_channels: int, feature_dim: int = 128):
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError("feature_dim must be divisible by 4 for InceptionTime.")
        out_channels = feature_dim // 4
        self.instance_encoder = nn.Sequential(
            InceptionBlock(n_in_channels, out_channels=out_channels, bottleneck_channels=out_channels),
            InceptionBlock(feature_dim, out_channels=out_channels, bottleneck_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_encoder(x)


def moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-1, -2) / (torch.max(col).clamp_min(1e-6) * torch.max(row).clamp_min(1e-6))
    eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    while eye.dim() < x.dim():
        eye = eye.unsqueeze(0)
    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * eye - (xz @ (15 * eye - (xz @ (7 * eye - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 16,
        heads: int = 8,
        num_landmarks: int = 64,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.residual = residual
        if residual:
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                kernel_size=(residual_conv_kernel, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    @staticmethod
    def _reshape_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
        b, n, hd = x.shape
        return x.view(b, n, heads, hd // heads).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def _landmarks(x: torch.Tensor, num_landmarks: int) -> torch.Tensor:
        b, h, n, d = x.shape
        m = min(num_landmarks, n)
        if n % m != 0:
            pad = m - (n % m)
            x = F.pad(x, (0, 0, 0, pad), value=0.0)
            n = x.shape[2]
        chunk = n // m
        return x.view(b, h, m, chunk, d).mean(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = self._reshape_heads(q, self.heads) * self.scale
        k = self._reshape_heads(k, self.heads)
        v = self._reshape_heads(v, self.heads)

        q_landmarks = self._landmarks(q, self.num_landmarks)
        k_landmarks = self._landmarks(k, self.num_landmarks)
        sim1 = torch.einsum("bhnd,bhmd->bhnm", q, k_landmarks)
        sim2 = torch.einsum("bhmd,bhkd->bhmk", q_landmarks, k_landmarks)
        sim3 = torch.einsum("bhmd,bhnd->bhmn", q_landmarks, k)

        attn1 = sim1.softmax(dim=-1)
        attn2_inv = moore_penrose_iter_pinv(sim2.softmax(dim=-1), self.pinv_iterations)
        attn3 = sim3.softmax(dim=-1)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)
        if self.residual:
            out = out + self.res_conv(v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class TransLayer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 8, dropout: float = 0.2, num_landmarks: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=max(1, dim // heads),
            heads=heads,
            num_landmarks=num_landmarks,
            pinv_iterations=6,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


def mexican_hat_wavelet(channels: int, kernel_size: int, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    x = torch.linspace(
        -(kernel_size - 1) // 2,
        (kernel_size - 1) // 2,
        kernel_size,
        dtype=scale.dtype,
        device=scale.device,
    )
    x = x.reshape(1, -1).repeat(channels, 1)
    scale = scale.reshape(channels, 1).abs().clamp_min(1e-3)
    shift = shift.reshape(channels, 1)
    x = x - shift
    c = 2.0 / (math.sqrt(3.0) * math.pi ** 0.25)
    return c * (1.0 - (x / scale) ** 2) * torch.exp(-((x / scale) ** 2) / 2.0) / torch.sqrt(scale)


class WaveletEncoding(nn.Module):
    def __init__(self, dim: int = 128, kernel_size: int = 19):
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, wave1: torch.Tensor, wave2: torch.Tensor, wave3: torch.Tensor) -> torch.Tensor:
        cls_token, feat_token = x[:, :1], x[:, 1:]
        feat = feat_token.transpose(1, 2)
        d = feat.shape[1]

        kernels = []
        for wave in (wave1, wave2, wave3):
            kernels.append(mexican_hat_wavelet(d, self.kernel_size, wave[0, :, 0], wave[1, :, 0]))
        pos = sum(F.conv1d(feat, kernel.unsqueeze(1), groups=d, padding=self.kernel_size // 2) for kernel in kernels)
        feat_token = feat_token + self.proj(pos.transpose(1, 2))
        return torch.cat((cls_token, feat_token), dim=1)


class Ref18_TimeMIL(nn.Module):
    def __init__(
        self,
        num_channels: int = 16,
        num_classes: int = 8,
        seq_length: int = 400,
        m_dim: int = 128,
        heads: int = 8,
        num_landmarks: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if m_dim % heads != 0:
            raise ValueError("m_dim must be divisible by heads")
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.m_dim = m_dim

        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=num_channels, feature_dim=m_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, m_dim))
        self.wave1 = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))
        self.wave2 = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))
        self.wave3 = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))
        self.wave1_ = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))
        self.wave2_ = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))
        self.wave3_ = nn.Parameter(torch.stack([torch.ones(m_dim, 1), torch.zeros(m_dim, 1)]))

        self.pos_layer = WaveletEncoding(m_dim)
        self.pos_layer2 = WaveletEncoding(m_dim)
        self.layer1 = TransLayer(dim=m_dim, heads=heads, dropout=dropout, num_landmarks=num_landmarks)
        self.layer2 = TransLayer(dim=m_dim, heads=heads, dropout=dropout, num_landmarks=num_landmarks)
        self.norm = nn.LayerNorm(m_dim)
        self.classifier = nn.Sequential(
            nn.Linear(m_dim, m_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(m_dim, num_classes),
        )
        initialize_weights(self)

    def forward(self, x: torch.Tensor, warmup: bool = False) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,C,T), got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_channels or x.shape[2] != self.seq_length:
            raise ValueError(f"Expected (B,{self.num_channels},{self.seq_length}), got {tuple(x.shape)}")

        x = self.feature_extractor(x).transpose(1, 2)  # (B, T, D)
        global_token = x.mean(dim=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)
        x = self.layer1(x)
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)
        x = self.layer2(x)
        x = self.norm(x)
        x = x[:, 0]
        if warmup:
            x = 0.1 * x + 0.99 * global_token
        return self.classifier(x)


if __name__ == "__main__":
    model = Ref18_TimeMIL()
    y = model(torch.randn(2, 16, 400))
    print(y.shape)
