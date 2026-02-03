import torch
import json
import shutil
import pickle
import lz4.frame
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.run_dir = config.base_dir / config.run_id
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, content, type='light'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cp_path = self.checkpoint_dir / f"checkpoint_{type}_{timestamp}"
        cp_path.mkdir(exist_ok=True)

        try:
            # Models (always saved)
            if content.get('atlas_model'):
                torch.save(content['atlas_model'], cp_path / 'atlas_model.pt')
            if content.get('entropy_model'):
                torch.save(content['entropy_model'], cp_path / 'entropy_model.pt')

            if type in ['light', 'full']:
                if content.get('atlas_opt'): torch.save(content['atlas_opt'], cp_path / 'atlas_opt.pt')
                if content.get('entropy_opt'): torch.save(content['entropy_opt'], cp_path / 'entropy_opt.pt')

                # RNG states
                rng_states = {
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state(),
                    'python': random.getstate()
                }
                if torch.cuda.is_available():
                    rng_states['cuda'] = torch.cuda.get_rng_state_all()
                torch.save(rng_states, cp_path / 'rng_states.pt')

                metadata = {
                    'atlas_step': content.get('atlas_step', 0),
                    'entropy_step': content.get('entropy_step', 0),
                    'atlas_games': content.get('atlas_games', 0),
                    'entropy_games': content.get('entropy_games', 0),
                    'timestamp': timestamp,
                    'type': type,
                    'config': vars(self.config) if hasattr(self.config, '__dict__') else {}
                }
                with open(cp_path / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

            if type == 'full':
                # Replay buffers
                if content.get('atlas_replay'):
                    with lz4.frame.open(cp_path / 'atlas_replay.lz4', 'wb') as f:
                        pickle.dump(content['atlas_replay'], f)
                if content.get('entropy_replay'):
                    with lz4.frame.open(cp_path / 'entropy_replay.lz4', 'wb') as f:
                        pickle.dump(content['entropy_replay'], f)

            # Pointer update
            with open(self.checkpoint_dir / 'latest_pointer.json', 'w') as f:
                json.dump({'path': str(cp_path), 'type': type}, f)

            # Rotation
            self._rotate_checkpoints()

            # If use_drive, copy to drive
            if self.config.use_drive:
                self._sync_to_drive(cp_path)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def _rotate_checkpoints(self):
        cps = sorted(list(self.checkpoint_dir.glob('checkpoint_*')), key=lambda x: x.stat().st_mtime)
        while len(cps) > self.config.checkpoint_keep_n:
            oldest = cps.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def _sync_to_drive(self, cp_path):
        drive_path = Path('/content/drive/MyDrive/prometheus_chess') / self.config.run_id / 'checkpoints' / cp_path.name
        drive_path.mkdir(parents=True, exist_ok=True)
        # Copy files (only models/metadata for light, everything for full)
        for f in cp_path.iterdir():
            shutil.copy2(f, drive_path / f.name)

    def load_latest(self):
        pointer_file = self.checkpoint_dir / 'latest_pointer.json'
        if not pointer_file.exists():
            return None

        with open(pointer_file, 'r') as f:
            pointer = json.load(f)

        cp_path = Path(pointer['path'])
        if not cp_path.exists():
            return None

        print(f"🔄 Loading checkpoint from {cp_path}")
        content = {}
        if (cp_path / 'atlas_model.pt').exists():
            content['atlas_model'] = torch.load(cp_path / 'atlas_model.pt', map_location='cpu', weights_only=False)
        if (cp_path / 'entropy_model.pt').exists():
            content['entropy_model'] = torch.load(cp_path / 'entropy_model.pt', map_location='cpu', weights_only=False)

        if (cp_path / 'atlas_opt.pt').exists():
            content['atlas_opt'] = torch.load(cp_path / 'atlas_opt.pt', map_location='cpu', weights_only=False)
        if (cp_path / 'entropy_opt.pt').exists():
            content['entropy_opt'] = torch.load(cp_path / 'entropy_opt.pt', map_location='cpu', weights_only=False)

        if (cp_path / 'rng_states.pt').exists():
            rng_states = torch.load(cp_path / 'rng_states.pt', map_location='cpu', weights_only=False)
            torch.set_rng_state(rng_states['torch'])
            np.random.set_state(rng_states['numpy'])
            random.setstate(rng_states['python'])
            if 'cuda' in rng_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states['cuda'])
            content['rng_states'] = rng_states

        if (cp_path / 'metadata.json').exists():
            with open(cp_path / 'metadata.json', 'r') as f:
                content['metadata'] = json.load(f)

        if (cp_path / 'atlas_replay.lz4').exists():
            with lz4.frame.open(cp_path / 'atlas_replay.lz4', 'rb') as f:
                content['atlas_replay'] = pickle.load(f)
        if (cp_path / 'entropy_replay.lz4').exists():
            with lz4.frame.open(cp_path / 'entropy_replay.lz4', 'rb') as f:
                content['entropy_replay'] = pickle.load(f)

        return content
