import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.atlas.network import ResidualBlock

class EntropyNetworkV2(nn.Module):
    """
    Vereinfachtes CNN-only Netzwerk für ENTROPY v2.0.

    Input: 22 Kanäle (19 Board + 3 Physik-Felder)
    Output:
    - Policy Logits: [batch, 4672]
    - Energy: [batch, 1] (unbeschränkt, aber mit Soft-Clipping)
    """

    def __init__(self, config: PrometheusConfig):
        super().__init__()
        C = config.entropy_channels

        # Input: 19 Board-Kanäle + 3 Physik-Feld-Kanäle
        self.input_block = nn.Sequential(
            nn.Conv2d(22, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU()
        )

        # Residual Tower (kleiner als ATLAS)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(C) for _ in range(config.entropy_res_blocks)
        ])

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, 4672)
        )

        # Energy Head
        self.energy_head = nn.Sequential(
            nn.Conv2d(C, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, board_tensor: torch.Tensor,
                field_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate board + fields
        x = torch.cat([board_tensor, field_tensor], dim=1)

        x = self.input_block(x)
        for block in self.res_blocks:
            x = block(x)

        policy_logits = self.policy_head(x)
        energy = self.energy_head(x)

        # Soft-Clipping für Energie-Stabilität: tanh(x/10)*10
        energy = torch.tanh(energy / 10.0) * 10.0

        return policy_logits, energy

    def get_features(self, board_tensor, field_tensor):
        """Returns the flattened features before heads, for RND"""
        x = torch.cat([board_tensor, field_tensor], dim=1)
        x = self.input_block(x)
        for block in self.res_blocks:
            x = block(x)
        return torch.flatten(x, 1)
