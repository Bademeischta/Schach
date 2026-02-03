import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDTarget(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        # Randomly initialized and frozen
        for p in self.parameters():
            p.requires_grad = False

class RNDPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.fc(x)

def compute_novelty(fused_features, rnd_target, rnd_predictor):
    with torch.no_grad():
        target_out = rnd_target(fused_features)
    predictor_out = rnd_predictor(fused_features)

    error = torch.mean((target_out - predictor_out)**2, dim=-1)
    return error
