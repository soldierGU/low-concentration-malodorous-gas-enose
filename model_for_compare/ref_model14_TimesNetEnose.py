import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def fft_for_period(x, k=2):
    """
    Safe version: run FFT on CPU to avoid CUDA/cuFFT UnicodeDecode issues on some Windows setups.
    x: (B, T, C) on any device
    return:
      period_list: (k,) on x.device (LongTensor)
      period_weight: (B, k) on x.device (FloatTensor)
    """
    device = x.device
    x_cpu = x.detach().to("cpu", non_blocking=False)

    xf = torch.fft.rfft(x_cpu, dim=1)  # CPU rFFT

    freq_amp = torch.abs(xf).mean(dim=0).mean(dim=-1)  # (F,)
    freq_amp[0] = 0.0

    top_idx = torch.topk(freq_amp, k=k, dim=0).indices  # (k,) on CPU
    T = x_cpu.size(1)
    top_idx_safe = torch.clamp(top_idx, min=1)
    period_list = (T // top_idx_safe).to(torch.long).to(device)

    period_weight = torch.abs(xf).mean(dim=-1).index_select(dim=1, index=top_idx)  # (B,k) CPU
    period_weight = period_weight.to(device)

    return period_list, period_weight


class Inception2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.b3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.merge = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y3 = self.b3(x)
        y5 = self.b5(x)
        return self.merge(torch.cat([y1, y3, y5], dim=1))


class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, top_k=2):
        super().__init__()
        self.seq_len = int(seq_len)
        self.top_k = int(top_k)
        self.conv = nn.Sequential(
            Inception2D(d_model, d_model),
            nn.GELU(),
            Inception2D(d_model, d_model),
        )

    def forward(self, x):
        B, T, C = x.shape
        period_list, period_weight = fft_for_period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = int(period_list[i].item())
            if period <= 0:
                period = 1

            if T % period != 0:
                length = ((T // period) + 1) * period
                out = torch.cat([x, x.new_zeros(B, length - T, C)], dim=1)
            else:
                length = T
                out = x

            out = out.reshape(B, length // period, period, C).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)
        w = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        out = torch.sum(res * w, dim=-1)
        return out + x


class Ref14_TimesNet_Enose(nn.Module):
    def __init__(self, num_classes, C_in=16, seq_len=400, d_model=64, e_layers=2, top_k=2, dropout=0.1):
        super().__init__()
        self.seq_len = int(seq_len)
        self.C_in = int(C_in)

        self.embedding = nn.Linear(self.C_in, d_model)
        self.blocks = nn.ModuleList([TimesBlock(self.seq_len, d_model, top_k=top_k) for _ in range(int(e_layers))])
        self.norm = nn.LayerNorm(d_model)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * self.seq_len, num_classes)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Expected input (B,C,T)")
        B, C, T = x.shape
        if C != self.C_in or T != self.seq_len:
            raise ValueError(f"Expected (C,T)=({self.C_in},{self.seq_len}), got ({C},{T})")

        x = x.transpose(1, 2)
        x = self.embedding(x)

        for blk in self.blocks:
            x = self.norm(blk(x))

        x = self.act(x)
        x = self.drop(x)
        x = x.reshape(B, -1)
        return self.projection(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Ref14_TimesNet_Enose(num_classes=8, C_in=16, seq_len=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output shape:", y.shape)
