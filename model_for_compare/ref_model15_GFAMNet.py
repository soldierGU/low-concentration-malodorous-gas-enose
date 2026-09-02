# ref_model15_GFAMNet.py
# GFAM-Net (Sensors & Actuators B: Chemical 396 (2023) 134551) adaptation for E-nose
# Input: (B, 16, 400)  i.e., (batch, sensors, time)
# Output: (B, num_classes)

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFAM(nn.Module):
    """
    Gas Feature Attention Mechanism (GFAM)
    Paper defines deep feature as X in R^{C x T x S}
    We implement X as tensor shape (B, C, T, S).

    Steps (paper):
      - UPeF / UInV / UStM: (B, C, 1, S)
      - depthwise conv over sensors with kernel (1, S) -> (B, C, 1, 1)
      - channel interaction via 1D conv (ECA style) with k=3 over channel axis
      - fuse: sigmoid(waPeF + waInV + waStM) -> w (B, C)
      - reweight X by w
    """
    def __init__(self, channels: int, sensors: int, eca_k: int = 3, stm_start: int = 51, stm_len: int = 10):
        super().__init__()
        self.channels = channels
        self.sensors = sensors
        self.stm_start = stm_start
        self.stm_len = stm_len

        # Depthwise conv to fuse per-sensor features for each channel: kernel (1, S)
        # Input: (B, C, 1, S) -> Output: (B, C, 1, 1)
        self.sensor_fuse = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, sensors),
            groups=channels,
            bias=False
        )

        # Channel Information Interaction (CII): ECA-style 1D conv across channel dimension
        # We treat channel descriptor as (B, 1, C) and apply conv1d(1->1,k=3)
        pad = (eca_k - 1) // 2
        self.cii_pef = nn.Conv1d(1, 1, kernel_size=eca_k, padding=pad, bias=False)
        self.cii_inv = nn.Conv1d(1, 1, kernel_size=eca_k, padding=pad, bias=False)
        self.cii_stm = nn.Conv1d(1, 1, kernel_size=eca_k, padding=pad, bias=False)

    @staticmethod
    def _pef(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Peak factor (PeF.)
        x: (B, C, T, S)
        returns: (B, C, 1, S)
        PeF = max|f(t)| / sqrt( sum(f(t)^2)/T )
        """
        # max over time
        peak = x.abs().amax(dim=2, keepdim=True)  # (B,C,1,S)
        rms = torch.sqrt((x.pow(2).mean(dim=2, keepdim=True)).clamp_min(eps))  # (B,C,1,S)
        return peak / rms

    @staticmethod
    def _inv(x: torch.Tensor) -> torch.Tensor:
        """
        Integral value (InV.)
        x: (B, C, T, S)
        returns: (B, C, 1, S)
        InV = ∫ f(t) dt  (discrete sum)
        """
        return x.sum(dim=2, keepdim=True)  # (B,C,1,S)

    def _stm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Steady mean (StM.)
        paper: StM = (1/10) sum_{t=t'}^{t'+9} f(t), with t'=51
        x: (B, C, T, S)
        returns: (B, C, 1, S)
        """
        B, C, T, S = x.shape
        t0 = min(max(self.stm_start, 0), max(T - 1, 0))
        t1 = min(t0 + self.stm_len, T)
        seg = x[:, :, t0:t1, :]  # (B,C,<=10,S)
        return seg.mean(dim=2, keepdim=True)  # (B,C,1,S)

    def _cii(self, u: torch.Tensor, conv1d: nn.Conv1d) -> torch.Tensor:
        """
        u: (B, C, 1, 1) after sensor fuse
        return: (B, C) channel weight logits after 1d conv across channels
        """
        # squeeze -> (B, C)
        u = u.squeeze(-1).squeeze(-1)  # (B,C)
        # ECA-style: (B, 1, C) -> conv1d -> (B, 1, C) -> (B, C)
        u = conv1d(u.unsqueeze(1)).squeeze(1)
        return u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, S)
        """
        # Step 1: compute attention descriptors per sensor
        u_pef = self._pef(x)          # (B,C,1,S)
        u_inv = self._inv(x)          # (B,C,1,S)
        u_stm = self._stm(x)          # (B,C,1,S)

        # Step 2: fuse sensor dimension -> (B,C,1,1)
        u_pef_f = self.sensor_fuse(u_pef)
        u_inv_f = self.sensor_fuse(u_inv)
        u_stm_f = self.sensor_fuse(u_stm)

        # Channel interaction (ECA-like)
        wa_pef = self._cii(u_pef_f, self.cii_pef)  # (B,C)
        wa_inv = self._cii(u_inv_f, self.cii_inv)  # (B,C)
        wa_stm = self._cii(u_stm_f, self.cii_stm)  # (B,C)

        # Step 3: fuse + sigmoid -> weights
        w = torch.sigmoid(wa_pef + wa_inv + wa_stm)  # (B,C)

        # reweight
        return x * w[:, :, None, None]


class Ref15_GFAMNet(nn.Module):
    """
    GFAM-Net adapted for E-nose input (B, 16, 400).
    We interpret:
      S = number of sensors = 16
      T = time length = 400
    We build feature map as (B, C_feat, T, S) using Conv2d.

    Network (as described):
      1) point conv to expand channels to 10
      2) depthwise strip conv (3x1) for feature extraction + GFAM
      3) avgpool (2x1) to compress along time
      4) FC(128) + classifier
    """
    def __init__(self, num_classes: int = 8, sensors: int = 16, time_len: int = 400,
                 feat_channels: int = 10, fc_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.sensors = sensors
        self.time_len = time_len
        self.feat_channels = feat_channels

        # Input reshape: (B,S,T) -> (B,1,T,S)
        # Pointwise conv: 1 -> feat_channels (paper uses 10-channel point conv) :contentReference[oaicite:2]{index=2}
        self.point_conv = nn.Sequential(
            nn.Conv2d(1, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # Depthwise strip conv: kernel 3x1 (retain temporal characteristics) :contentReference[oaicite:3]{index=3}
        self.strip_dw = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=(3, 1),
                      padding=(1, 0), groups=feat_channels, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.gfam = GFAM(channels=feat_channels, sensors=sensors, eca_k=3, stm_start=51, stm_len=10)

        # AvgPool (2x1) :contentReference[oaicite:4]{index=4}
        self.pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        # compute flattened dim after pool
        t_after = time_len // 2  # stride=2 once
        flat_dim = feat_channels * t_after * sensors

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(fc_dim, num_classes)

        # Xavier init (paper mentions Xavier for conv & FC) :contentReference[oaicite:5]{index=5}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, T)  e.g. (B,16,400)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input (B,S,T), got shape {tuple(x.shape)}")
        B, S, T = x.shape
        if S != self.sensors or T != self.time_len:
            raise ValueError(f"Expected (S,T)=({self.sensors},{self.time_len}), got ({S},{T})")

        # (B,S,T) -> (B,1,T,S)
        x = x.permute(0, 2, 1).unsqueeze(1)

        x = self.point_conv(x)   # (B,Cf,T,S)
        x = self.strip_dw(x)     # (B,Cf,T,S)
        x = self.gfam(x)         # (B,Cf,T,S)
        x = self.pool(x)         # (B,Cf,T/2,S)

        x = x.flatten(1)         # (B, flat_dim)
        x = self.fc(x)           # (B, 128)
        x = self.classifier(x)   # (B, num_classes)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Ref15_GFAMNet(num_classes=8, sensors=16, time_len=400).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2,8)
