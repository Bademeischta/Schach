import json
import os
from pathlib import Path

class MetricsLogger:
    def __init__(self, run_dir):
        self.metrics_dir = run_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / 'metrics.jsonl'
        self.f = open(self.metrics_file, 'a', buffering=1)

    def log(self, data):
        data['timestamp'] = os.times().elapsed
        self.f.write(json.dumps(data) + '\n')

    def close(self):
        self.f.close()
