import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from prometheus_tqfd.config import PrometheusConfig

class EntropyV2Loss:
    """
    Hybrid Loss-System für ENTROPY v2.0.
    """

    def __init__(self, config: PrometheusConfig, device: str):
        self.config = config
        self.device = device
        self.weights = {
            'outcome': config.entropy_loss_outcome,
            'mobility': config.entropy_loss_mobility,
            'pressure': config.entropy_loss_pressure,
            'stability': config.entropy_loss_stability,
            'novelty': config.entropy_loss_novelty,
        }

        # RND Networks
        # Input features are C*64, where C is entropy_channels (128) -> 8192
        # Wait, get_features returns flattened res_tower output.
        # C=128, kernel 3, padding 1 keeps 8x8. So 128*8*8 = 8192.
        self.feature_dim = config.entropy_channels * 64
        self.rnd_target = self._make_rnd_net(frozen=True).to(device)
        self.rnd_predictor = self._make_rnd_net(frozen=False).to(device)

    def _make_rnd_net(self, frozen: bool) -> nn.Module:
        net = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        if frozen:
            for param in net.parameters():
                param.requires_grad = False
        return net

    def compute(self, batch: Dict, game_results: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Berechnet den Gesamtverlust.
        """
        losses = {}

        # 1. Outcome Loss (sparse)
        losses['outcome'] = self._outcome_loss(batch['energy'], game_results)

        # 2. Mobility Loss
        losses['mobility'] = self._mobility_loss(batch['policy_logits'], batch['legal_counts_self'])

        # 3. Pressure Loss
        losses['pressure'] = self._pressure_loss(batch['legal_counts_self'], batch['legal_counts_opponent'])

        # 4. Stability Loss (TD)
        losses['stability'] = self._stability_loss(batch['energy'], batch['energy_next'])

        # 5. Novelty Loss (RND)
        losses['novelty'] = self._novelty_loss(batch['features'])

        # Gewichtete Summe
        total = sum(self.weights[k] * losses[k] for k in losses)

        return total, {k: v.item() for k, v in losses.items()}

    def _outcome_loss(self, energy: torch.Tensor, results: torch.Tensor) -> torch.Tensor:
        target = results.float().view(-1, 1) * 5.0  # Skalierung
        return F.smooth_l1_loss(energy, target)

    def _mobility_loss(self, policy_logits: torch.Tensor, legal_counts: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(policy_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

        # Normalisiere mit Anzahl legaler Züge
        max_entropy = torch.log(legal_counts.float() + 1e-8)
        normalized_entropy = entropy / (max_entropy + 1e-8)

        return -normalized_entropy.mean()

    def _pressure_loss(self, our_legal: torch.Tensor, opp_legal: torch.Tensor) -> torch.Tensor:
        ratio = opp_legal.float() / (our_legal.float() + 1.0)
        return ratio.mean()

    def _stability_loss(self, energy_now: torch.Tensor, energy_next: torch.Tensor) -> torch.Tensor:
        gamma = 0.99
        target = gamma * energy_next.detach()
        return F.mse_loss(energy_now, target)

    def _novelty_loss(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_out = self.rnd_target(features)
        predictor_out = self.rnd_predictor(features)

        error = ((target_out - predictor_out) ** 2).mean(dim=1)
        return error.mean()
