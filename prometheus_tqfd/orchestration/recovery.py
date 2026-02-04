import torch
import gc
import time
from multiprocessing import Event
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.orchestration.checkpoint import CheckpointManager

class OOMHandler:
    """
    Behandelt Out-of-Memory Situationen.
    """

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.current_batch_size_atlas = config.atlas_batch_size
        self.current_batch_size_entropy = config.entropy_batch_size
        self.oom_count = 0

    def handle_oom(self, process_name: str, checkpoint_manager: CheckpointManager,
                   shared_values: dict, pause_event: Event):
        self.oom_count += 1
        print(f"⚠️ OOM #{self.oom_count} in {process_name}")

        # 1. Pause setzen
        pause_event.set()

        # 2. GPU-Speicher freigeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 3. Batch-Size reduzieren
        self.current_batch_size_atlas = max(
            self.config.oom_min_batch_size,
            int(self.current_batch_size_atlas * self.config.oom_batch_reduction)
        )
        self.current_batch_size_entropy = max(
            self.config.oom_min_batch_size,
            int(self.current_batch_size_entropy * self.config.oom_batch_reduction)
        )

        print(f"   Reduced batch sizes: ATLAS={self.current_batch_size_atlas}, ENTROPY={self.current_batch_size_entropy}")

        # 4. Wait
        time.sleep(5)

        # 5. Load latest checkpoint to shared state
        checkpoint = checkpoint_manager.load_latest()
        if checkpoint:
            shared_values.update(checkpoint)

        # 6. Resume
        pause_event.clear()
        print(f"✅ System resumed after OOM.")
