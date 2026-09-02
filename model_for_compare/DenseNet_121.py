import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock1D(nn.Module):
    def __init__(self, input_channels, growth_rate, num_layers):
        super(DenseBlock1D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    # Bottleneck层（1×1卷积）
                    nn.Conv1d(input_channels + i * growth_rate, 4 * growth_rate, kernel_size=1, bias=False),
                    nn.BatchNorm1d(4 * growth_rate),
                    nn.ReLU(inplace=True),
                    # 3×3卷积
                    nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(growth_rate),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class Transition1D(nn.Module):
    def __init__(self, input_channels, compression_rate=0.5):
        super(Transition1D, self).__init__()
        self.conv = nn.Conv1d(input_channels, int(input_channels * compression_rate), kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(int(input_channels * compression_rate))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DenseNet1D_121(nn.Module):
    def __init__(self, num_classes=8, num_channels=16, growth_rate=32, compression_rate=0.5):
        super(DenseNet1D_121, self).__init__()
        self.growth_rate = int(growth_rate)
        self.compression_rate = float(compression_rate)
        self.num_layers_per_block = [6, 12, 24, 16]  # DenseNet-121 的结构

        # 初始卷积层
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Dense Block 1
        self.dense_block1 = self._make_dense_block(64, self.num_layers_per_block[0])
        channels_after_block1 = 64 + self.num_layers_per_block[0] * self.growth_rate
        self.transition1 = Transition1D(channels_after_block1, compression_rate=self.compression_rate)

        # Dense Block 2
        channels_after_transition1 = int(channels_after_block1 * self.compression_rate)
        self.dense_block2 = self._make_dense_block(channels_after_transition1, self.num_layers_per_block[1])
        channels_after_block2 = channels_after_transition1 + self.num_layers_per_block[1] * self.growth_rate
        self.transition2 = Transition1D(channels_after_block2, compression_rate=self.compression_rate)

        # Dense Block 3
        channels_after_transition2 = int(channels_after_block2 * self.compression_rate)
        self.dense_block3 = self._make_dense_block(channels_after_transition2, self.num_layers_per_block[2])
        channels_after_block3 = channels_after_transition2 + self.num_layers_per_block[2] * self.growth_rate
        self.transition3 = Transition1D(channels_after_block3, compression_rate=self.compression_rate)

        # Dense Block 4
        channels_after_transition3 = int(channels_after_block3 * self.compression_rate)
        self.dense_block4 = self._make_dense_block(channels_after_transition3, self.num_layers_per_block[3])
        channels_after_block4 = channels_after_transition3 + self.num_layers_per_block[3] * self.growth_rate

        # Final BatchNorm and Classifier
        self.final_bn = nn.BatchNorm1d(channels_after_block4)
        self.fc = nn.Linear(channels_after_block4, num_classes)

    def _make_dense_block(self, input_channels, num_layers):
        return DenseBlock1D(input_channels, self.growth_rate, num_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.dense_block1(x)
        x = self.transition1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.dense_block3(x)
        x = self.transition3(x)

        x = self.dense_block4(x)

        x = self.final_bn(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
