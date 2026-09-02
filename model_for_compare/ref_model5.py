import torch
import torch.nn as nn
import torch.nn.functional as F
# A novel DenseNet with warm restarts for gas recognition in complex airflow environments
# Microchemical Journal
# 1D-DNR

class Bottleneck1D(nn.Module):
    """
    Bottleneck layer for 1D-DenseNet
    """

    def __init__(self, in_channels, growth_rate):
        super(Bottleneck1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = torch.cat([x, out], 1)  # Concatenate input and output along channel dimension
        return out


class Transition1D(nn.Module):
    """
    Transition layer for 1D-DenseNet
    """

    def __init__(self, in_channels, out_channels):
        super(Transition1D, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.bn(x))
        out = self.conv(out)
        out = self.pool(out)
        return out


class DenseBlock1D(nn.Module):
    """
    Dense block for 1D-DenseNet
    """

    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock1D, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck1D(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class OneD_DNR(nn.Module):
    """
    1D-DenseNet with warm restarts for gas recognition
    """

    def __init__(self, num_classes=8, growth_rate=16, block_layers=[6, 12, 24]):
        super(OneD_DNR, self).__init__()
        # 修改初始卷积层：输入通道从 1 改为 16
        self.conv1 = nn.Conv1d(16, 12, kernel_size=3, padding=1)

        self.dense_block1 = DenseBlock1D(12, growth_rate, block_layers[0])
        self.trans1 = Transition1D(12 + block_layers[0] * growth_rate, (12 + block_layers[0] * growth_rate) // 2)

        self.dense_block2 = DenseBlock1D((12 + block_layers[0] * growth_rate) // 2, growth_rate, block_layers[1])
        self.trans2 = Transition1D((12 + block_layers[0] * growth_rate) // 2 + block_layers[1] * growth_rate,
                                   ((12 + block_layers[0] * growth_rate) // 2 + block_layers[1] * growth_rate) // 2)

        self.dense_block3 = DenseBlock1D(
            ((12 + block_layers[0] * growth_rate) // 2 + block_layers[1] * growth_rate) // 2,
            growth_rate, block_layers[2])

        # 修改全连接层输出维度：从 10 改为 8
        self.bn = nn.BatchNorm1d(((12 + block_layers[0] * growth_rate) // 2 + block_layers[1] * growth_rate) // 2 +
                                 block_layers[2] * growth_rate)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(((12 + block_layers[0] * growth_rate) // 2 + block_layers[1] * growth_rate) // 2 +
                            block_layers[2] * growth_rate, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense_block1(out)
        out = self.trans1(out)
        out = self.dense_block2(out)
        out = self.trans2(out)
        out = self.dense_block3(out)
        out = self.bn(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 示例用法
if __name__ == "__main__":
    model = OneD_DNR(num_classes=8)  # 8类
    dummy_input = torch.randn(1, 16, 400)  # 输入形状 (batch_size, channels=16, length=400)
    output = model(dummy_input)
    print(output.shape)  # 应输出 (1, 8)
