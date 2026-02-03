import torch
import gc
import time
import psutil

class RecoveryManager:
    def __init__(self, config, stop_event, oom_event):
        self.config = config
        self.stop_event = stop_event
        self.oom_event = oom_event
        self.oom_count = 0

    def handle_oom(self, process_name):
        print(f"⚠️ OOM detected in {process_name}")
        self.oom_event.set()

        # 1. Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # 2. Adjust config (reduce batch sizes)
        if self.config.atlas_batch_size > self.config.oom_min_batch_size:
            self.config.atlas_batch_size = int(self.config.atlas_batch_size * self.config.oom_batch_reduction_factor)
            print(f"Reduced ATLAS batch size to {self.config.atlas_batch_size}")

        if self.config.entropy_batch_size > self.config.oom_min_batch_size:
            self.config.entropy_batch_size = int(self.config.entropy_batch_size * self.config.oom_batch_reduction_factor)
            print(f"Reduced ENTROPY batch size to {self.config.entropy_batch_size}")

        if self.config.atlas_mcts_simulations > 50:
            self.config.atlas_mcts_simulations //= 2
            print(f"Reduced MCTS simulations to {self.config.atlas_mcts_simulations}")

        self.oom_count += 1
        time.sleep(5)
        self.oom_event.clear()

    def check_system_health(self):
        # RAM check
        ram_usage = psutil.virtual_memory().percent
        if ram_usage > 90:
            print(f"⚠️ High RAM usage: {ram_usage}%")
            # Possible action: clear replay buffers or restart

        # GPU Temp check (if pynvml is available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp > 85:
                print(f"⚠️ High GPU Temp: {temp}°C")
                return "hot"
        except:
            pass
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
