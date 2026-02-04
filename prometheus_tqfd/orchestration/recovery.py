import torch
import gc
import time
import psutil
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

class RecoveryManager:
    def __init__(self, config, stop_event, oom_event):
        self.config = config
        self.stop_event = stop_event
        self.oom_event = oom_event
        self.oom_count = 0

    def handle_oom(self, process_name):
        print(f"⚠️ OOM detected in {process_name}")
        self.oom_event.set()
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)
        self.oom_event.clear()

    def check_system_health(self):
        ram_usage = psutil.virtual_memory().percent
        if ram_usage > 90:
            return "hot"
        return "ok"

def guarded_run(target_fn, name, recovery_mgr, *args, **kwargs):
    restarts = 0
    max_restarts = 3
    while restarts < max_restarts and not recovery_mgr.stop_event.is_set():
        try:
            target_fn(*args, **kwargs)
            break
        except torch.cuda.OutOfMemoryError:
            recovery_mgr.handle_oom(name)
            restarts += 1
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            restarts += 1
            time.sleep(5)

    if restarts >= max_restarts:
        print(f"Process {name} failed after {max_restarts} restarts.")
        recovery_mgr.stop_event.set()
