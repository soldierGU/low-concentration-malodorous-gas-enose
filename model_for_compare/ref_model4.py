import torch
import torch.nn as nn
import torch.nn.functional as F


# Anti-Drift Gas Detection Algorithm Based on Neural Network
#MCNN


class SelfAttention(nn.Module):
    def __init__(self, in_channels, in_length):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.in_length = in_length
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, L = x.size()

        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, -1, L).permute(0, 2, 1)  # B x L x C'
        key = self.key_conv(x).view(batch_size, -1, L)  # B x C' x L
        value = self.value_conv(x).view(batch_size, -1, L)  # B x C x L

        # Compute attention scores
        attention_scores = torch.bmm(query, key)  # B x L x L
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Compute weighted value
        weighted_value = torch.bmm(value, attention_scores.permute(0, 2, 1))  # B x C x L

        # Combine with original input
        out = self.gamma * weighted_value + x
        return out


class MultiscaleCNN(nn.Module):
    def __init__(self, num_classes, num_channels=16, in_length=400):
        super(MultiscaleCNN, self).__init__()
        self.attention = SelfAttention(num_channels, in_length)

        # Multiscale convolutional layers
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Define the output length calculation function
        def calculate_output_length(input_length, kernel_size, stride=1, padding=0, dilation=1):
            return (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        # Initial input length after attention
        initial_length = in_length

        # Compute output lengths for each branch
        conv1_out_len_before_pool = calculate_output_length(initial_length, kernel_size=1)
        conv1_out_len_after_pool = calculate_output_length(conv1_out_len_before_pool, kernel_size=2, stride=2)


        conv2_out_len_before_pool = calculate_output_length(initial_length, kernel_size=2)
        conv2_out_len_after_pool = calculate_output_length(conv2_out_len_before_pool, kernel_size=2, stride=2)

        conv3_out_len_before_pool = calculate_output_length(initial_length, kernel_size=3)
        conv3_out_len_after_pool = calculate_output_length(conv3_out_len_before_pool, kernel_size=2, stride=2)

        # Total number of features
        total_features = 64 * (conv1_out_len_after_pool + conv2_out_len_after_pool + conv3_out_len_after_pool)

        # Fully connected layers
        self.fc1 = nn.Linear(total_features, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply attention mechanism
        x = self.attention(x)

        # Apply multiscale convolutions
        conv1_out = self.pool(F.relu(self.conv1(x)))
        conv2_out = self.pool(F.relu(self.conv2(x)))
        conv3_out = self.pool(F.relu(self.conv3(x)))

        # Flatten and concatenate multiscale features
        conv1_out = conv1_out.view(conv1_out.size(0), -1)
        conv2_out = conv2_out.view(conv2_out.size(0), -1)
        conv3_out = conv3_out.view(conv3_out.size(0), -1)

        combined_features = torch.cat((conv1_out, conv2_out, conv3_out), dim=1)

        # Print the shape of combined_features for debugging
        # print("Combined features shape:", combined_features.shape)

        # Fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)

        return x


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = MultiscaleCNN(num_classes=8)  # 8 gas classes

    # Dummy input (batch_size=32, channels=16, length=400)
    dummy_input = torch.randn(32, 16, 400)

    # Forward pass
    output = model(dummy_input)
    print(output.shape)  # Should be (32, 8)