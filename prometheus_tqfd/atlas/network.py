import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class PolicyHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, 4672)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        return self.fc(out)

class ValueHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 4, kernel_size=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        return torch.tanh(self.fc2(out))

class AtlasNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(config.atlas_input_channels, config.atlas_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.atlas_channels),
            nn.ReLU()
        )
        self.res_tower = nn.ModuleList([
            ResidualBlock(config.atlas_channels) for _ in range(config.atlas_res_blocks)
        ])
        self.policy_head = PolicyHead(config.atlas_channels)
        self.value_head = ValueHead(config.atlas_channels)

    def forward(self, x):
        x = self.input_block(x)
        for block in self.res_tower:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
