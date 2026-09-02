import torch
import torch.nn as nn
import torch.nn.functional as F
# Toward Accurate Odor Identification and Effective Feature Learning With an AI-Empowered Electronic Nose
# CNN-AE
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=10, stride=1):
        super(ConvBlock, self).__init__()
        # 保持时间步不变（kernel_size=3, padding=1）
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 降采样：每次时间步减少 1/3
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        return x


class OneDCNN_article1(nn.Module):
    def __init__(self, num_channels=16, num_classes=8, embedding_size=128):
        super(OneDCNN_article1, self).__init__()
        # 4 个 ConvBlock，每个保持通道数 128
        self.conv_block1 = ConvBlock(num_channels, 128)
        self.conv_block2 = ConvBlock(128, 128)
        self.conv_block3 = ConvBlock(128, 128)
        self.conv_block4 = ConvBlock(128, 128)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 嵌入层（可选）
        self.fc_to_embedding = nn.Linear(128, embedding_size)
        # 分类器
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # 输入形状: (batch_size, 16, 400)
        x = self.conv_block1(x)  # (batch_size, 128, 400/3 = 133)
        x = self.conv_block2(x)  # (batch_size, 128, 133/3 = 44)
        x = self.conv_block3(x)  # (batch_size, 128, 44/3 = 14)
        x = self.conv_block4(x)  # (batch_size, 128, 14/3 = 4)

        # 全局平均池化
        x = self.global_avg_pool(x)  # (batch_size, 128, 1)
        x = x.squeeze(-1)  # (batch_size, 128)

        # 可选嵌入层（可删除以直接连接分类器）
        # embedding = self.fc_to_embedding(x)
        # outputs = self.classifier(embedding)

        # 直接连接分类器
        outputs = self.classifier(x)
        return outputs
