import torch
import json
import shutil
import pickle
import lz4.frame
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
        # content: dict with models, optimizers, etc.
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cp_path = self.checkpoint_dir / f"checkpoint_{type}_{timestamp}"
        cp_path.mkdir(exist_ok=True)

        try:
            # Models (always saved)
            torch.save(content['atlas_model'], cp_path / 'atlas_model.pt')
            torch.save(content['entropy_model'], cp_path / 'entropy_model.pt')

            if type in ['light', 'full']:
                torch.save(content['atlas_opt'], cp_path / 'atlas_opt.pt')
                torch.save(content['entropy_opt'], cp_path / 'entropy_opt.pt')
                torch.save(content['rng_states'], cp_path / 'rng_states.pt')

                metadata = {
                    'atlas_step': content['atlas_step'],
                    'entropy_step': content['entropy_step'],
                    'atlas_games': content['atlas_games'],
                    'entropy_games': content['entropy_games'],
                    'timestamp': timestamp,
                    'type': type
                }
                with open(cp_path / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

            if type == 'full':
                # Replay buffers
                with lz4.frame.open(cp_path / 'atlas_replay.lz4', 'wb') as f:
                    pickle.dump(content['atlas_replay'], f)
                with lz4.frame.open(cp_path / 'entropy_replay.lz4', 'wb') as f:
                    pickle.dump(content['entropy_replay'], f)

            # Update latest
            latest_link = self.checkpoint_dir / 'latest'
            if latest_link.exists():
                if latest_link.is_symlink(): latest_link.unlink()
                else: shutil.rmtree(latest_link)

            # Since symlinks can be tricky in some environments (like Colab/Drive),
            # we might just copy the latest or use a pointer file.
            # I'll use a pointer file.
            with open(self.checkpoint_dir / 'latest_pointer.json', 'w') as f:
                json.dump({'path': str(cp_path), 'type': type}, f)

            # If use_drive, copy to drive
            if self.config.use_drive:
                self._sync_to_drive(cp_path)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

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

        content = {}
        content['atlas_model'] = torch.load(cp_path / 'atlas_model.pt', map_location='cpu')
        content['entropy_model'] = torch.load(cp_path / 'entropy_model.pt', map_location='cpu')

        if (cp_path / 'atlas_opt.pt').exists():
            content['atlas_opt'] = torch.load(cp_path / 'atlas_opt.pt', map_location='cpu')
            content['entropy_opt'] = torch.load(cp_path / 'entropy_opt.pt', map_location='cpu')
            content['rng_states'] = torch.load(cp_path / 'rng_states.pt', map_location='cpu')
            with open(cp_path / 'metadata.json', 'r') as f:
                content['metadata'] = json.load(f)

        if (cp_path / 'atlas_replay.lz4').exists():
            with lz4.frame.open(cp_path / 'atlas_replay.lz4', 'rb') as f:
                content['atlas_replay'] = pickle.load(f)
            with lz4.frame.open(cp_path / 'entropy_replay.lz4', 'rb') as f:
                content['entropy_replay'] = pickle.load(f)

        return content
