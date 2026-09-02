import torch
import torch.nn as nn
import torch.nn.functional as F
# 效果非常好

class Inception1D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception1D, self).__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm1d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv followed by 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm1d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 conv followed by 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm1d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch5x5red, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pooling followed by 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate along the channel dimension
        outputs = [branch1, branch2, branch3, branch4]
        outputs = torch.cat(outputs, 1)

        return outputs


class GoogLeNet1D(nn.Module):
    def __init__(self, num_channels=5, num_classes=8):
        super(GoogLeNet1D, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Inception modules
        self.inception3a = Inception1D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception1D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception1D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception1D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception1D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception1D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception1D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception1D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception1D(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x