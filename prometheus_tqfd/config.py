from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import torch
import os

@dataclass
class EntropyPhysicsConfig:
    # Figuren-Energien
    energy_king: float = 1000.0
    energy_queen: float = 9.5
    energy_rook: float = 5.25
    energy_bishop: float = 3.33
    energy_knight: float = 3.05
    energy_pawn: float = 1.0

    # Feld-Parameter
    diffusion_sigma: float = 2.5
    field_alpha: float = 1.0  # Gewicht Masse-Feld
    field_beta: float = 0.5   # Gewicht Druck-Feld
    field_gamma: float = 0.1  # Gewicht Laplacian

    # Thermodynamik
    boltzmann_temp_start: float = 5.0
    boltzmann_temp_end: float = 0.1
    jarzynski_beta_start: float = 0.1
    jarzynski_beta_end: float = 2.0

@dataclass
class PrometheusConfig:
    # === SYSTEM ===
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    base_dir: Path = Path('./prometheus_runs')
    use_drive: bool = False
    seed: int = 42

    # === HARDWARE ===
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_device: str = 'cpu' # For stability in Colab
    num_atlas_actors: int = 2
    num_entropy_actors: int = 2

    # === ATLAS ===
    atlas_input_channels: int = 19
    atlas_res_blocks: int = 10
    atlas_channels: int = 256
    atlas_learning_rate: float = 1e-3
    atlas_weight_decay: float = 1e-4
    atlas_batch_size: int = 256
    atlas_replay_size: int = 500_000
    atlas_mcts_simulations: int = 200
    atlas_mcts_cpuct_base: float = 19652.0
    atlas_mcts_cpuct_init: float = 2.5
    atlas_dirichlet_alpha: float = 0.3
    atlas_dirichlet_epsilon: float = 0.25
    atlas_temp_moves: int = 30
    atlas_temp_init: float = 1.0
    atlas_temp_final: float = 0.1

    # === ENTROPY ===
    entropy_gnn_hidden: int = 64
    entropy_gnn_heads: int = 4
    entropy_gnn_layers: int = 4
    entropy_field_channels: int = 64
    entropy_fusion_dim: int = 512
    entropy_learning_rate: float = 3e-4
    entropy_batch_size: int = 128
    entropy_replay_size: int = 200_000
    entropy_temp_start: float = 5.0
    entropy_temp_end: float = 0.1
    entropy_temp_decay: float = 0.9999
    entropy_beta_start: float = 0.1
    entropy_beta_end: float = 2.0
    entropy_loss_weight_conservation: float = 0.5
    entropy_loss_weight_smoothness: float = 0.1
    entropy_loss_weight_novelty: float = 0.3
    entropy_rnd_feature_dim: int = 128

    physics: EntropyPhysicsConfig = field(default_factory=EntropyPhysicsConfig)

    # === TRAINING ===
    min_buffer_before_training: int = 10_000
    weight_publish_interval: int = 100  # steps
    checkpoint_interval_minutes: int = 15
    checkpoint_keep_n: int = 5

    # === EVALUATION ===
    eval_interval_games: int = 10_000
    eval_games_per_duel: int = 100
    eval_time_per_move: float = 5.0
    elo_initial: float = 1000.0
    elo_k_factor: float = 32.0

    # === DASHBOARD ===
    dashboard_port: int = 8501
    dashboard_refresh_seconds: float = 2.0

    # === RESILIENZ ===
    heartbeat_timeout: float = 60.0
    max_process_restarts: int = 3
    oom_batch_reduction_factor: float = 0.5
    oom_min_batch_size: int = 8

def adjust_config_for_hardware(config: PrometheusConfig):
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.atlas_batch_size = 32
        config.entropy_batch_size = 16
        config.atlas_res_blocks = 5
        config.atlas_mcts_simulations = 50
        return config

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if vram_gb >= 40:  # A100
        config.atlas_batch_size = 512
        config.entropy_batch_size = 256
        config.atlas_mcts_simulations = 400
        config.atlas_res_blocks = 15
        config.atlas_replay_size = 1_000_000
    elif vram_gb >= 16:  # V100 / T4-Pro
        config.atlas_batch_size = 256
        config.entropy_batch_size = 128
        config.atlas_mcts_simulations = 200
        config.atlas_res_blocks = 10
        config.atlas_replay_size = 500_000
    elif vram_gb >= 12:  # T4-Free
        config.atlas_batch_size = 128
        config.entropy_batch_size = 64
        config.atlas_mcts_simulations = 100
        config.atlas_res_blocks = 8
        config.atlas_replay_size = 300_000
    else:
        config.atlas_batch_size = 64
        config.entropy_batch_size = 32
        config.atlas_mcts_simulations = 50
        config.atlas_res_blocks = 5
        config.atlas_replay_size = 100_000

    return config
