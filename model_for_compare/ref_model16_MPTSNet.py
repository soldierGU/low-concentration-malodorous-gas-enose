# -*- coding: utf-8 -*-
"""MPTSNet adaptation for the local e-nose comparison pipeline.

Reference: https://github.com/MUYang99/MPTSNet
Input:  (B, 16, 400)
Output: (B, num_classes)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_length: int):
        super().__init__()
        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = 0 if kernel_size == 1 else kernel_size // 2
        padding_mode = "zeros" if kernel_size == 1 else "circular"
        self.token_conv = nn.Conv1d(
            c_in,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, embed_dim: int, seq_length: int, dropout: float = 0.1, pointwise: bool = False):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, embed_dim, kernel_size=1 if pointwise else 3)
        self.position_embedding = PositionalEmbedding(embed_dim, max_len=max(seq_length, 20000 if pointwise else seq_length))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class InceptionCBAM(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.kernels = nn.ModuleList(
            [nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i) for i in range(num_kernels)]
        )
        self.cbam = CBAMBlock(out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)
        return self.cbam(res)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)
        return self.layernorm2(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self, length: int, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, length)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_encoding(x.permute(2, 0, 1))
        for block in self.blocks:
            x = block(x)
        return x.permute(1, 2, 0).contiguous()


class WindowTransformer(nn.Module):
    def __init__(self, in_channels: int, length: int, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Linear(in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, length)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x.permute(0, 2, 1))
        x = self.positional_encoding(x.permute(1, 0, 2))
        for block in self.blocks:
            x = block(x)
        x = x.permute(1, 0, 2).contiguous()
        return self.output_layer(x).permute(0, 2, 1)


def fft_main_periods_wo_duplicates(data: torch.Tensor, k: int) -> list[int]:
    """Find dominant periods from training tensors shaped (N, C, T)."""
    x = data.detach().float()
    x = x.permute(0, 2, 1)  # (N, T, C)
    n = x.shape[1]
    spectrum = torch.fft.rfft(x, dim=1).abs().mean(dim=0).mean(dim=-1)
    spectrum[0] = 0
    if spectrum.numel() > 1:
        spectrum[1] = 0
    order = torch.argsort(spectrum, descending=True)
    periods = []
    used = set()
    for idx in order.tolist():
        if idx <= 0:
            continue
        period = max(1, int(round(n / idx)))
        if period not in used:
            used.add(period)
            periods.append(period)
        if len(periods) == k:
            break
    return periods or [n]


def fft_find_each_amplitude_torch(x: torch.Tensor, target_period: int) -> torch.Tensor:
    """Per-sample amplitude used for period fusion. Input is (B, C, T).

    The reference implementation computes this branch on detached CPU numpy
    arrays. Keeping that behavior avoids CUDA FFT runtime issues on Windows.
    """
    data = x.detach().float().cpu().numpy()
    batch_size = data.shape[0]
    sequence_length = data.shape[2]
    target_frequency = sequence_length / max(1, target_period)
    xf = np.fft.fftfreq(sequence_length, 1.0 / sequence_length)[: sequence_length // 2]
    amplitudes = torch.zeros((batch_size, 1), dtype=x.dtype)
    for i in range(batch_size):
        averaged_data = data[i].mean(axis=0)
        yf = np.fft.fft(averaged_data)
        power_spectrum = 2.0 / sequence_length * np.abs(yf[: sequence_length // 2])
        closest_index = int(np.argmin(np.abs(xf - target_frequency)))
        amplitudes[i, 0] = float(power_spectrum[closest_index])
    return amplitudes.to(x.device)


class PeriodicBlock(nn.Module):
    def __init__(
        self,
        periods: list[int],
        seq_length: int,
        embed_dim: int,
        embed_dim_t: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        cnn_hidden: int = 1024,
    ):
        super().__init__()
        self.periods = [max(1, int(p)) for p in periods]
        self.embed_dim = embed_dim
        self.time_transformer = Transformer(seq_length, embed_dim, num_heads, ff_dim, num_layers)
        self.cnn = nn.Sequential(InceptionCBAM(embed_dim, cnn_hidden), nn.GELU(), InceptionCBAM(cnn_hidden, embed_dim))
        self.window_transformers = nn.ModuleList(
            [
                WindowTransformer(embed_dim * period, (seq_length // period) + 1, embed_dim_t, num_heads, ff_dim, num_layers)
                for period in self.periods
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        time_point_features = self.time_transformer(x)

        global_features = []
        amplitudes = []
        for period, transformer in zip(self.periods, self.window_transformers):
            amplitudes.append(fft_find_each_amplitude_torch(x, period))
            if t % period != 0:
                length = ((t // period) + 1) * period
                padding = torch.zeros(b, c, length - t, device=x.device, dtype=x.dtype)
                out = torch.cat([x, padding], dim=2)
            else:
                length = t
                out = x
            num_period = length // period
            out = out.reshape(b, c, period, num_period).contiguous()
            window_batch = out.permute(0, 3, 1, 2).reshape(b * num_period, c, period)
            local_features = self.cnn(window_batch)
            local_features = local_features.reshape(b, num_period, c, period).permute(0, 2, 3, 1).contiguous()
            local_features = out + local_features
            local_features = local_features.reshape(b, -1, num_period)
            global_feature = transformer(local_features)
            global_feature = global_feature.reshape(b, self.embed_dim, -1).contiguous()[:, :, :t]
            global_features.append(global_feature)

        weights = torch.softmax(torch.cat(amplitudes, dim=1), dim=1)
        stacked = torch.stack(global_features, dim=-1)
        period_weight = weights[:, None, None, :].repeat(1, self.embed_dim, t, 1)
        return torch.sum(stacked * period_weight, dim=-1) + time_point_features + x


class Ref16_MPTSNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        num_channels: int = 16,
        seq_length: int = 400,
        periods: list[int] | tuple[int, ...] | None = None,
        embed_dim: int | None = None,
        embed_dim_t: int | None = None,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 1,
        num_blocks: int = 2,
        cnn_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.periods = list(periods) if periods is not None else [400, 200, 100, 80, 50]
        embed_dim = embed_dim or max(min(num_channels * 4, 256), 64)
        embed_dim_t = embed_dim_t or max(min(embed_dim * 4, 512), 256)

        self.enc_embedding = DataEmbedding(num_channels, embed_dim, seq_length, dropout=dropout, pointwise=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList(
            [
                PeriodicBlock(self.periods, seq_length, embed_dim, embed_dim_t, num_heads, ff_dim, num_layers, cnn_hidden)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_length * embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,C,T), got {tuple(x.shape)}")
        x = self.enc_embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.blocks:
            x = self.layer_norm(block(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(F.gelu(x))
        return self.fc(x.reshape(x.shape[0], -1).float())


def build_mptsnet_from_training_data(
    train_x: torch.Tensor,
    num_classes: int = 8,
    num_channels: int = 16,
    seq_length: int = 400,
    top_k_periods: int = 5,
) -> Ref16_MPTSNet:
    periods = fft_main_periods_wo_duplicates(train_x, top_k_periods)
    return Ref16_MPTSNet(num_classes=num_classes, num_channels=num_channels, seq_length=seq_length, periods=periods)


if __name__ == "__main__":
    model = Ref16_MPTSNet()
    y = model(torch.randn(2, 16, 400))
    print(y.shape)
