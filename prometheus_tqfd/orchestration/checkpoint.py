import torch
import json
import time
import random
import numpy as np
import pickle
import lz4.frame
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from prometheus_tqfd.config import PrometheusConfig

class CheckpointManager:
    """
    Tiered Checkpoint-System: Micro (5min), Light (15min), Full (60min).
    """

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.base_path = config.base_dir / config.run_id / 'checkpoints'
        self.base_path.mkdir(parents=True, exist_ok=True)

        # drive path if used
        self.drive_path = Path('/content/drive/MyDrive/prometheus_chess') / config.run_id / 'checkpoints' if config.use_drive else None
        if self.drive_path:
            self.drive_path.mkdir(parents=True, exist_ok=True)

        self.last_micro = time.time()
        self.last_light = time.time()
        self.last_full = time.time()

    def maybe_checkpoint(self, shared_values: dict, replay_buffers: dict = None):
        now = time.time()

        if now - self.last_full > self.config.checkpoint_full_interval * 60:
            self.save_full(shared_values, replay_buffers)
            self.last_full = now
            self.last_light = now
            self.last_micro = now
        elif now - self.last_light > self.config.checkpoint_light_interval * 60:
            self.save_light(shared_values)
            self.last_light = now
            self.last_micro = now
        elif now - self.last_micro > self.config.checkpoint_micro_interval * 60:
            self.save_micro(shared_values)
            self.last_micro = now

    def save_micro(self, shared_values: dict):
        path = self.base_path / 'latest'
        path.mkdir(exist_ok=True)

        if 'atlas_weights' in shared_values:
            torch.save(shared_values['atlas_weights'], path / 'atlas_weights.pt')
        if 'entropy_weights' in shared_values:
            torch.save(shared_values['entropy_weights'], path / 'entropy_weights.pt')

        metadata = {
            'type': 'micro',
            'timestamp': datetime.now().isoformat(),
            'atlas_steps': shared_values.get('atlas_steps', 0),
            'entropy_steps': shared_values.get('entropy_steps', 0),
            'atlas_version': shared_values.get('atlas_version', 0),
            'entropy_version': shared_values.get('entropy_version', 0),
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    def save_light(self, shared_values: dict):
        self.save_micro(shared_values)
        path = self.base_path / 'latest'

        if 'atlas_optimizer' in shared_values:
            torch.save(shared_values['atlas_optimizer'], path / 'atlas_optimizer.pt')
        if 'entropy_optimizer' in shared_values:
            torch.save(shared_values['entropy_optimizer'], path / 'entropy_optimizer.pt')

        rng_states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states['cuda'] = torch.cuda.get_rng_state()
        torch.save(rng_states, path / 'rng_states.pt')

        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        metadata['type'] = 'light'
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    def save_full(self, shared_values: dict, replay_buffers: dict = None):
        self.save_light(shared_values)
        path = self.base_path / 'latest'

        if replay_buffers:
            for name, buffer in replay_buffers.items():
                with lz4.frame.open(path / f'{name}_replay.lz4', 'wb') as f:
                    pickle.dump(buffer.get_data(), f)

        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        metadata['type'] = 'full'
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Archive current 'latest' to a timestamped folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_path = self.base_path / f'full_{timestamp}'
        import shutil
        shutil.copytree(path, archive_path)

        # Sync to drive if enabled
        if self.drive_path:
            drive_latest = self.drive_path / 'latest'
            if drive_latest.exists(): shutil.rmtree(drive_latest)
            shutil.copytree(path, drive_latest)
            print(f"💾 Checkpoint mirrored to Google Drive")

    def load_latest(self) -> Optional[dict]:
        path = self.base_path / 'latest'
        # If not local, check drive
        if not path.exists() and self.drive_path:
            drive_path = self.drive_path / 'latest'
            if drive_path.exists():
                import shutil
                shutil.copytree(drive_path, path)
                print("📂 Restored checkpoint from Google Drive")

        if not path.exists():
            return None

        data = {}
        try:
            if (path / 'atlas_weights.pt').exists():
                data['atlas_weights'] = torch.load(path / 'atlas_weights.pt')
            if (path / 'entropy_weights.pt').exists():
                data['entropy_weights'] = torch.load(path / 'entropy_weights.pt')
            if (path / 'atlas_optimizer.pt').exists():
                data['atlas_optimizer'] = torch.load(path / 'atlas_optimizer.pt')
            if (path / 'entropy_optimizer.pt').exists():
                data['entropy_optimizer'] = torch.load(path / 'entropy_optimizer.pt')
            if (path / 'rng_states.pt').exists():
                data['rng_states'] = torch.load(path / 'rng_states.pt')
            if (path / 'metadata.json').exists():
                with open(path / 'metadata.json', 'r') as f:
                    data['metadata'] = json.load(f)
            return data
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            return None

    def load_replay_buffer(self, name: str) -> Optional[list]:
        path = self.base_path / 'latest' / f'{name}_replay.lz4'
        if path.exists():
            with lz4.frame.open(path, 'rb') as f:
                return pickle.load(f)
        return None
