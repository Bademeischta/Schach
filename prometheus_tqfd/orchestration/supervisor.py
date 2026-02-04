import time
import os
import signal
import multiprocessing as mp
from typing import Dict, List
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.orchestration.checkpoint import CheckpointManager
from prometheus_tqfd.orchestration.recovery import OOMHandler, guarded_run
from prometheus_tqfd.evaluation.arena import Arena

# --- Standalone Runner Functions to avoid pickling 'self' ---

def run_atlas_trainer(config, data_queue, weights_queue, stop_event, pause_event, gpu_lock, heartbeats, metrics_queue, shared_values, device):
    from prometheus_tqfd.atlas.trainer import AtlasTrainer
    from prometheus_tqfd.orchestration.checkpoint import CheckpointManager
    from prometheus_tqfd.orchestration.recovery import OOMHandler

    trainer = AtlasTrainer(config, data_queue, weights_queue, device, shared_values)
    checkpoint_manager = CheckpointManager(config)
    oom_handler = OOMHandler(config)

    try:
        trainer.run(stop_event, pause_event, gpu_lock, heartbeats, metrics_queue)
    except Exception as e:
        if "out of memory" in str(e).lower():
            oom_handler.handle_oom('atlas_trainer', checkpoint_manager, shared_values, pause_event)
        else:
            raise e

def run_entropy_trainer(config, data_queue, weights_queue, stop_event, pause_event, gpu_lock, heartbeats, metrics_queue, shared_values, device):
    from prometheus_tqfd.entropy.trainer import EntropyTrainer
    from prometheus_tqfd.orchestration.checkpoint import CheckpointManager
    from prometheus_tqfd.orchestration.recovery import OOMHandler

    trainer = EntropyTrainer(config, data_queue, weights_queue, device, shared_values)
    checkpoint_manager = CheckpointManager(config)
    oom_handler = OOMHandler(config)

    try:
        trainer.run(stop_event, pause_event, gpu_lock, heartbeats, metrics_queue)
    except Exception as e:
        if "out of memory" in str(e).lower():
            oom_handler.handle_oom('entropy_trainer', checkpoint_manager, shared_values, pause_event)
        else:
            raise e

def run_atlas_selfplay(config, weights_queue, data_queue, stop_event, heartbeats, worker_id, device, metrics_queue):
    from prometheus_tqfd.atlas.selfplay import AtlasSelfPlayWorker
    worker = AtlasSelfPlayWorker(config, weights_queue, data_queue, device, worker_id)
    worker.run(stop_event, heartbeats, metrics_queue)

def run_entropy_selfplay(config, weights_queue, data_queue, stop_event, heartbeats, worker_id, device, metrics_queue):
    from prometheus_tqfd.entropy.selfplay import EntropySelfPlayWorker
    worker = EntropySelfPlayWorker(config, weights_queue, data_queue, device, worker_id)
    worker.run(stop_event, heartbeats, metrics_queue)

class Supervisor:
    """
    Hauptorchestrierer des Systems.
    """

    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.device = 'cuda' if torch_is_cuda() else 'cpu'

        # Shared State
        self.manager = mp.Manager()
        self.shared_values = self.manager.dict()
        self.heartbeats = self.manager.dict()

        # Events
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        self.oom_event = mp.Event() # Added as in recovery.py guarded_run logic

        # Locks
        self.gpu_lock = mp.Lock()

        # Queues
        self.atlas_data_queue = mp.Queue(maxsize=100)
        self.atlas_weights_queue = mp.Queue(maxsize=1)
        self.entropy_data_queue = mp.Queue(maxsize=100)
        self.entropy_weights_queue = mp.Queue(maxsize=1)
        self.metrics_queue = mp.Queue(maxsize=10000)

        # Components
        self.checkpoint_manager = CheckpointManager(config)
        self.oom_handler = OOMHandler(config)
        self.arena = Arena(config)
        from prometheus_tqfd.utils.logging import MetricsLogger
        self.metrics_logger = MetricsLogger(config)

        from prometheus_tqfd.orchestration.recovery import RecoveryManager
        self.recovery_mgr = RecoveryManager(config, self.stop_event, self.oom_event)

        # Processes
        self.processes = {}

    def start(self):
        """Startet alle Prozesse"""
        # Trainers
        self._start_process('atlas_trainer', run_atlas_trainer, (
            self.config, self.atlas_data_queue, self.atlas_weights_queue,
            self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats,
            self.metrics_queue, self.shared_values, self.device
        ))

        self._start_process('entropy_trainer', run_entropy_trainer, (
            self.config, self.entropy_data_queue, self.entropy_weights_queue,
            self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats,
            self.metrics_queue, self.shared_values, self.device
        ))

        # Self-Play Workers
        for i in range(self.config.num_atlas_selfplay_workers):
            self._start_process(f'atlas_selfplay_{i}', run_atlas_selfplay, (
                self.config, self.atlas_weights_queue, self.atlas_data_queue,
                self.stop_event, self.heartbeats, i, 'cpu', self.metrics_queue
            ))

        for i in range(self.config.num_entropy_selfplay_workers):
            self._start_process(f'entropy_selfplay_{i}', run_entropy_selfplay, (
                self.config, self.entropy_weights_queue, self.entropy_data_queue,
                self.stop_event, self.heartbeats, i, 'cpu', self.metrics_queue
            ))

    def run(self):
        """Hauptschleife"""
        self.start()
        last_eval = 0

        try:
            while not self.stop_event.is_set():
                # 1. Heartbeats prüfen
                self._check_heartbeats()

                # 2. Metrics sammeln und loggen
                self._collect_metrics()

                # 3. Checkpoints
                self.checkpoint_manager.maybe_checkpoint(dict(self.shared_values))

                # 4. Evaluation
                atlas_steps = self.shared_values.get('atlas_steps', 0)
                entropy_steps = self.shared_values.get('entropy_steps', 0)
                total_steps = atlas_steps + entropy_steps

                if total_steps - last_eval >= self.config.eval_interval_games:
                    self._run_evaluation()
                    last_eval = total_steps

                time.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Supervisor: Shutdown requested...")
        finally:
            self.stop()

    def stop(self):
        self.stop_event.set()
        print("💾 Saving final checkpoint...")
        self.checkpoint_manager.save_full(dict(self.shared_values))

        print("🧹 Terminating processes...")
        for name, proc in self.processes.items():
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        print("✅ Supervisor: Shutdown complete.")

    def _start_process(self, name: str, target_fn, args):
        # Wrap target_fn with guarded_run
        p = mp.Process(target=guarded_run, args=(target_fn, name, self.recovery_mgr) + args, name=name, daemon=True)
        p.start()
        self.processes[name] = p
        print(f"🚀 Started {name} (PID: {p.pid})")

    def _check_heartbeats(self):
        now = time.time()
        for name, last_beat in dict(self.heartbeats).items():
            if now - last_beat > self.config.heartbeat_timeout:
                print(f"⚠️ {name} timed out, restarting...")
                self._restart_process(name)

    def _collect_metrics(self):
        """Drains the metrics queue and logs to file."""
        try:
            while not self.metrics_queue.empty():
                m = self.metrics_queue.get_nowait()
                self.metrics_logger.log(m)
        except:
            pass

    def _run_evaluation(self):
        print("⚔️ Running Arena Evaluation...")
        from prometheus_tqfd.atlas.network import AtlasNetwork
        from prometheus_tqfd.entropy.network import EntropyNetworkV2

        atlas_net = AtlasNetwork(self.config).to('cpu')
        entropy_net = EntropyNetworkV2(self.config).to('cpu')

        if 'atlas_weights' in self.shared_values:
            atlas_net.load_state_dict(self.shared_values['atlas_weights'])
        if 'entropy_weights' in self.shared_values:
            entropy_net.load_state_dict(self.shared_values['entropy_weights'])

        results = self.arena.run_evaluation(atlas_net, entropy_net, 'cpu')
        results['type'] = 'evaluation'
        self.metrics_logger.log(results)

        # Update shared ELO
        for k, v in results['elo'].items():
            self.shared_values[f'elo_{k}'] = v
        print(f"📊 New ELOs: {results['elo']}")

    def _restart_process(self, name: str):
        if name in self.processes:
            p = self.processes[name]
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        if 'atlas_trainer' in name:
            self._start_process(name, run_atlas_trainer, (
                self.config, self.atlas_data_queue, self.atlas_weights_queue,
                self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats,
                self.metrics_queue, self.shared_values, self.device
            ))
        elif 'entropy_trainer' in name:
            self._start_process(name, run_entropy_trainer, (
                self.config, self.entropy_data_queue, self.entropy_weights_queue,
                self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats,
                self.metrics_queue, self.shared_values, self.device
            ))
        elif 'atlas_selfplay' in name:
            i = int(name.split('_')[-1])
            self._start_process(name, run_atlas_selfplay, (
                self.config, self.atlas_weights_queue, self.atlas_data_queue,
                self.stop_event, self.heartbeats, i, 'cpu', self.metrics_queue
            ))
        elif 'entropy_selfplay' in name:
            i = int(name.split('_')[-1])
            self._start_process(name, run_entropy_selfplay, (
                self.config, self.entropy_weights_queue, self.entropy_data_queue,
                self.stop_event, self.heartbeats, i, 'cpu', self.metrics_queue
            ))

def torch_is_cuda():
    import torch
    return torch.cuda.is_available()
