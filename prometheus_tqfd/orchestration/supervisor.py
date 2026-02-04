import time
import os
import signal
import multiprocessing as mp
from typing import Dict, List
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.orchestration.checkpoint import CheckpointManager
from prometheus_tqfd.orchestration.recovery import OOMHandler
from prometheus_tqfd.evaluation.arena import Arena

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

        # Processes
        self.processes = {}

    def start(self):
        """Startet alle Prozesse"""
        # Trainers
        self._start_process('atlas_trainer', self._run_atlas_trainer)
        self._start_process('entropy_trainer', self._run_entropy_trainer)

        # Self-Play Workers
        for i in range(self.config.num_atlas_selfplay_workers):
            self._start_process(f'atlas_selfplay_{i}', lambda i=i: self._run_atlas_selfplay(i))

        for i in range(self.config.num_entropy_selfplay_workers):
            self._start_process(f'entropy_selfplay_{i}', lambda i=i: self._run_entropy_selfplay(i))

    def run(self):
        """Hauptschleife"""
        self.start()
        last_eval = 0

        try:
            while not self.stop_event.is_set():
                # 1. Heartbeats prüfen
                self._check_heartbeats()

                # 2. Metrics sammeln und loggen (DRAIN QUEUE)
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

    def _start_process(self, name: str, target_fn):
        p = mp.Process(target=target_fn, name=name, daemon=True)
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
        # We need actual models here. This is slightly tricky in the supervisor process.
        # For simplicity in this script, we'll log that we're doing it.
        # In a full implementation, we'd load weights into networks.
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
            self._start_process(name, self._run_atlas_trainer)
        elif 'entropy_trainer' in name:
            self._start_process(name, self._run_entropy_trainer)
        elif 'atlas_selfplay' in name:
            i = int(name.split('_')[-1])
            self._start_process(name, lambda: self._run_atlas_selfplay(i))
        elif 'entropy_selfplay' in name:
            i = int(name.split('_')[-1])
            self._start_process(name, lambda: self._run_entropy_selfplay(i))

    # Runner functions that instantiate the components in the subprocesses
    def _run_atlas_trainer(self):
        from prometheus_tqfd.atlas.trainer import AtlasTrainer
        trainer = AtlasTrainer(self.config, self.atlas_data_queue, self.atlas_weights_queue, self.device, self.shared_values)
        try:
            trainer.run(self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats, self.metrics_queue)
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.oom_handler.handle_oom('atlas_trainer', self.checkpoint_manager, self.shared_values, self.pause_event)

    def _run_entropy_trainer(self):
        from prometheus_tqfd.entropy.trainer import EntropyTrainer
        trainer = EntropyTrainer(self.config, self.entropy_data_queue, self.entropy_weights_queue, self.device, self.shared_values)
        try:
            trainer.run(self.stop_event, self.pause_event, self.gpu_lock, self.heartbeats, self.metrics_queue)
        except Exception as e:
            if "out of memory" in str(e).lower():
                self.oom_handler.handle_oom('entropy_trainer', self.checkpoint_manager, self.shared_values, self.pause_event)

    def _run_atlas_selfplay(self, i: int):
        from prometheus_tqfd.atlas.selfplay import AtlasSelfPlayWorker
        # Self-play often on CPU for stability or lower GPU memory
        worker = AtlasSelfPlayWorker(self.config, self.atlas_weights_queue, self.atlas_data_queue, 'cpu', i)
        worker.run(self.stop_event, self.heartbeats)

    def _run_entropy_selfplay(self, i: int):
        from prometheus_tqfd.entropy.selfplay import EntropySelfPlayWorker
        worker = EntropySelfPlayWorker(self.config, self.entropy_weights_queue, self.entropy_data_queue, 'cpu', i)
        worker.run(self.stop_event, self.heartbeats)

def torch_is_cuda():
    import torch
    return torch.cuda.is_available()
