import torch
import torch.nn as nn
import torch.nn.functional as F

# Origin identification of Angelica dahurica using a bidirectional mixing  network combined with an electronic nose system


class MLP1D(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act = act()

    def forward(self, x):
        # x: (..., in_dim)
        return self.fc2(self.act(self.fc1(x)))

class BMM(nn.Module):
    """
    输入/输出: (B, C, T)
    """
    def __init__(self, T, C, K=20):
        super().__init__()
        self.T, self.C = T, C

        # time local: conv over T (3x1) on (T,C) grid
        self.time_conv = nn.Conv2d(1, K, kernel_size=(3, 1), padding=(1, 0))
        self.time_pw   = nn.Conv2d(K, 1, kernel_size=1)
        self.time_mlp  = MLP1D(in_dim=T, hidden_dim=T // 2, out_dim=T)

        # sensor local: conv over C (1x3)
        self.sens_conv = nn.Conv2d(1, K, kernel_size=(1, 3), padding=(0, 1))
        self.sens_pw   = nn.Conv2d(K, 1, kernel_size=1)
        self.sens_mlp  = MLP1D(in_dim=C, hidden_dim=max(1, C // 2), out_dim=C)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        assert C == self.C and T == self.T, f"Got (C,T)=({C},{T}), expected ({self.C},{self.T})"

        # -> (B,1,T,C)
        x0 = x.permute(0, 2, 1).unsqueeze(1)

        # ----- Time correlation -----
        t = F.relu(self.time_conv(x0))   # (B,K,T,C)
        t = F.relu(self.time_pw(t))      # (B,1,T,C)

        # time global MLP: per-sensor, last dim must be T
        t_in  = t.squeeze(1).permute(0, 2, 1)   # (B,C,T)  ✅ last dim = T
        t_out = self.time_mlp(t_in)             # (B,C,T)
        t2    = t_out.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,C)

        xQ = x0 + t2

        # ----- Sensor correlation -----
        s = F.relu(self.sens_conv(xQ))   # (B,K,T,C)
        s = F.relu(self.sens_pw(s))      # (B,1,T,C)

        # sensor global MLP: per-time, last dim must be C
        s_in  = s.squeeze(1)             # (B,T,C) ✅ last dim = C
        s_out = self.sens_mlp(s_in)      # (B,T,C)
        s2    = s_out.unsqueeze(1)       # (B,1,T,C)

        xhat = xQ + s2

        # -> (B,C,T)
        return xhat.squeeze(1).permute(0, 2, 1)

class BMNet(nn.Module):
    def __init__(self, T, C, num_classes, K=20):
        super().__init__()
        self.bmm  = BMM(T=T, C=C, K=K)
        self.pool = nn.AvgPool1d(kernel_size=2)     # over T
        self.fc1  = nn.Linear(C * (T // 2), 64)
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B,C,T)
        feat = self.bmm(x)          # (B,C,T)
        feat = self.pool(feat)      # (B,C,T/2)
        feat = feat.flatten(1)      # (B, C*T/2)
        h = torch.sigmoid(self.fc1(feat))
        return self.fc2(h)


# quick sanity check
if __name__ == "__main__":
    model = BMNet(T=400, C=16, num_classes=8)
    x = torch.randn(8, 16, 400)   # (B, C, T)
    y = model(x)
    print(y.shape)  # torch.Size([8, 4])
