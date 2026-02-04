import json
import time
from pathlib import Path

class MetricsLogger:
    def __init__(self, config):
        self.config = config
        self.log_file = config.base_dir / config.run_id / 'metrics' / 'metrics.jsonl'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics):
        metrics['timestamp'] = time.time()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
