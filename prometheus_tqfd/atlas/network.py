import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from prometheus_tqfd.config import PrometheusConfig

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x

class AtlasNetwork(nn.Module):
    """
    Standard AlphaZero-Architektur:
    - Input: [batch, 19, 8, 8]
    - Residual Tower: N Blöcke mit Skip-Connections
    - Policy Head: Wahrscheinlichkeiten über 4672 Züge
    - Value Head: Gewinnwahrscheinlichkeit [-1, +1]
    """

    def __init__(self, config: PrometheusConfig):
        super().__init__()
        C = config.atlas_channels

        # Input Block
        self.input_block = nn.Sequential(
            nn.Conv2d(19, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )

        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(C)
            for _ in range(config.atlas_res_blocks)
        ])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, 4672)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(C, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_block(x)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
