import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.dsc = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        identity = x  # 保存输入作为残差连接的输入
        out = self.relu1(x)  # 第一层 ReLU
        out = self.dsc(out)  # 深度可分离卷积
        if self.shortcut:  # 调整输入维度以匹配输出维度
            identity = self.shortcut(identity)
        out += identity  # 将残差连接到卷积块的输出
        out = self.relu2(out)  # 第二层 ReLU
        out = F.max_pool1d(out, kernel_size=2, stride=2)  # 最大池化
        return out


class LZM_1DCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(LZM_1DCNN, self).__init__()
        self.initial_conv = DepthwiseSeparableConv(16, 16, kernel_size=3, stride=1, padding=1)

        # 3个16通道输入的卷积块
        self.block1_1 = ResidualBlock(16, 16)
        self.block1_2 = ResidualBlock(16, 16)
        self.block1_3 = ResidualBlock(16, 16)

        # 3个32通道输入的卷积块
        self.block2_1 = ResidualBlock(16, 32, stride=2)  # 下采样
        self.block2_2 = ResidualBlock(32, 32)
        self.block2_3 = ResidualBlock(32, 32)

        # 1个64通道输入的卷积块
        self.block3_1 = ResidualBlock(32, 64, stride=2)  # 下采样

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.block1_3(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block2_3(x)
        x = self.block3_1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LZM_1DCNN(num_classes=8).to(device)
    x = torch.randn(2, 16, 400, device=device)
    y = model(x)
    print("Output:", y.shape)  # (2, 8)
