import torch
import torch.nn.functional as F

def entropy_loss(policy_self, board_batch):
    # S_self = -Σ p log p
    s_self = -torch.sum(policy_self * torch.log(policy_self + 1e-8), dim=-1)

    # S_opponent approximation: log(num_legal)
    # We need num_legal_opponent for each board in batch
    # This might be slow if we calculate it here, but let's assume it's passed
    pass

def compute_total_entropy_loss(batch_data, model, rnd_target, rnd_predictor, config):
    # batch_data contains: field_tensors, board_data, legal_masks, opponent_legal_counts, etc.
    probs, energy, flow, (real, imag), fused = model(batch_data['fields'], batch_data['boards'], batch_data['masks'])

    # 1. Entropy Ratio Loss
    s_self = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    s_opp = torch.log(batch_data['opp_legal_counts'].float() + 1.0)
    l_ent = -torch.log(s_self / (s_opp + 0.1) + 1e-8).mean()

    # 2. Conservation Loss: |ΔE - E_captured|²
    # Need e_before, e_after, e_captured
    # This requires sequential data or pairs of states
    l_cons = F.mse_loss(batch_data['energy_diff'], -batch_data['energy_captured'])

    # 3. Smoothness Loss: |∇²Φ|²
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).float().to(energy.device).view(1, 1, 3, 3)
    # Φ is part of fields, let's extract it
    phi = batch_data['fields'][:, 3:4, :, :]
    laplacian = F.conv2d(phi, laplacian_kernel, padding=1)
    l_smooth = (laplacian**2).mean()

    # 4. TD Loss (Energy)
    # E(s) vs γ E(s')
    target_energy = config.entropy_beta_start * batch_data['energy_next'].detach()
    l_td = F.mse_loss(energy, target_energy)

    # 5. Novelty Loss
    rnd_error = torch.mean((rnd_target(fused) - rnd_predictor(fused))**2, dim=-1)
    l_novel = rnd_error.mean()

    total_loss = (
        1.0 * l_ent +
        config.entropy_loss_weight_conservation * l_cons +
        config.entropy_loss_weight_smoothness * l_smooth +
        1.0 * l_td +
        config.entropy_loss_weight_novelty * l_novel
    )

    return total_loss, l_ent.item(), l_cons.item(), l_smooth.item(), l_td.item(), l_novel.item()
