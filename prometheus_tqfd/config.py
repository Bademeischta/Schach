from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import torch
import os
from typing import Optional, Tuple

@dataclass
class PrometheusConfig:
    # === SYSTEM ===
    run_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    base_dir: Path = Path('/content/prometheus_runs')
    use_drive: bool = True
    seed: int = 42

    # === ATLAS (AlphaZero-Stil) ===
    atlas_input_channels: int = 19
    atlas_res_blocks: int = 8              # Reduziert für Colab
    atlas_channels: int = 256
    atlas_learning_rate: float = 1e-3
    atlas_weight_decay: float = 1e-4
    atlas_batch_size: int = 128            # Hardware-adaptiv
    atlas_replay_size: int = 300_000
    atlas_mcts_simulations: int = 100
    atlas_mcts_cpuct: float = 2.5
    atlas_dirichlet_alpha: float = 0.3
    atlas_dirichlet_epsilon: float = 0.25
    atlas_temperature_moves: int = 30
    atlas_temperature_init: float = 1.0
    atlas_temperature_final: float = 0.1

    # === ENTROPY v2.0 (Hybrid Physik) ===
    entropy_input_channels: int = 22       # 19 Board + 3 Felder
    entropy_res_blocks: int = 6
    entropy_channels: int = 128
    entropy_learning_rate: float = 3e-4
    entropy_batch_size: int = 64
    entropy_replay_size: int = 200_000

    # Mini-Rollout Parameter
    entropy_rollout_depth: int = 3
    entropy_rollout_count: int = 5
    entropy_temperature_start: float = 3.0
    entropy_temperature_end: float = 0.3
    entropy_temperature_decay: float = 0.9999

    # Loss-Gewichtung (muss 1.0 ergeben)
    entropy_loss_outcome: float = 0.30     # Spielergebnis
    entropy_loss_mobility: float = 0.25    # Eigene Optionen
    entropy_loss_pressure: float = 0.20    # Druck auf Gegner
    entropy_loss_stability: float = 0.15   # TD auf Energie
    entropy_loss_novelty: float = 0.10     # RND Exploration

    # Physik-Konstanten
    physics_energy_king: float = 1000.0
    physics_energy_queen: float = 9.5
    physics_energy_rook: float = 5.25
    physics_energy_bishop: float = 3.33
    physics_energy_knight: float = 3.05
    physics_energy_pawn: float = 1.0
    physics_diffusion_sigma: float = 2.5
    physics_field_alpha: float = 1.0
    physics_field_beta: float = 0.5

    # === TACTICS DETECTOR ===
    tactics_boost_strength: float = 1.0      # Volle Stärke
    tactics_boost_decay: float = 1.0         # Kein Decay (permanent)
    tactics_mate_boost: float = 50.0         # Boost für Matt-in-1
    tactics_hanging_boost: float = 5.0       # Boost für Figurenrettung
    tactics_threat_boost: float = 3.0        # Boost für Verteidigung

    # === TRAINING ===
    min_buffer_before_training: int = 5_000
    weight_publish_interval: int = 50
    gpu_priority_atlas: float = 0.6        # 60% Priorität für ATLAS
    num_atlas_selfplay_workers: int = 1
    num_entropy_selfplay_workers: int = 1

    # === CHECKPOINTS ===
    checkpoint_micro_interval: int = 5     # Minuten
    checkpoint_light_interval: int = 15    # Minuten
    checkpoint_full_interval: int = 60     # Minuten
    checkpoint_keep_n: int = 3

    # === EVALUATION ===
    eval_interval_games: int = 5_000
    eval_games_atlas_entropy: int = 50
    eval_games_vs_random: int = 20
    eval_games_vs_heuristic: int = 20
    elo_initial: float = 1000.0
    elo_k_factor: float = 32.0

    # === DASHBOARD ===
    dashboard_port: int = 8501
    dashboard_refresh_seconds: float = 2.0

    # === RESILIENZ ===
    heartbeat_timeout: float = 60.0
    max_process_restarts: int = 3
    oom_batch_reduction: float = 0.5
    oom_min_batch_size: int = 8

@dataclass
class HardwareConfig:
    device: str
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    ram_gb: float
    cpu_cores: int
    is_colab: bool

def adjust_config_for_hardware(config: PrometheusConfig, hw: HardwareConfig) -> PrometheusConfig:
    """
    Erkennt Hardware und setzt optimale Parameter.
    """
    if hw.device == 'cpu':
        config.atlas_batch_size = 64
        config.entropy_batch_size = 32
        config.atlas_res_blocks = 5
        config.atlas_mcts_simulations = 50
        config.atlas_replay_size = 100_000
        return config

    vram_gb = hw.vram_gb or 0.0

    if vram_gb >= 40:  # A100
        config.atlas_batch_size = 1024
        config.atlas_mcts_simulations = 800
        config.atlas_res_blocks = 15
        config.atlas_replay_size = 2_000_000
        config.entropy_batch_size = 512
        config.entropy_replay_size = 1_000_000
    elif vram_gb >= 16:  # V100 / T4-Pro
        config.atlas_batch_size = 512
        config.atlas_mcts_simulations = 400
        config.atlas_res_blocks = 12
        config.atlas_replay_size = 1_000_000
        config.entropy_batch_size = 256
        config.entropy_replay_size = 500_000
    elif vram_gb >= 12:  # T4-Free
        config.atlas_batch_size = 512
        config.atlas_mcts_simulations = 200
        config.atlas_res_blocks = 10
        config.atlas_replay_size = 500_000
        config.entropy_batch_size = 256
        config.entropy_replay_size = 300_000
    else:  # < 8GB (z.B. K80 oder kleine lokale GPU)
        config.atlas_batch_size = 128
        config.atlas_mcts_simulations = 100
        config.atlas_res_blocks = 5
        config.atlas_replay_size = 100_000
        config.entropy_batch_size = 32
        config.entropy_replay_size = 100_000

    # Worker-Anzahl anpassen
    if hw.cpu_cores >= 8:
        config.num_atlas_selfplay_workers = 2
        config.num_entropy_selfplay_workers = 2
    else:
        config.num_atlas_selfplay_workers = 1
        config.num_entropy_selfplay_workers = 1

    return config
