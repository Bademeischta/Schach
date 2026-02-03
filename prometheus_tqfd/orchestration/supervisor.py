import torch.multiprocessing as mp
import time
import random
import torch
from prometheus_tqfd.orchestration.checkpoint import CheckpointManager
from prometheus_tqfd.orchestration.recovery import RecoveryManager, guarded_run
from prometheus_tqfd.atlas.trainer import AtlasTrainer
from prometheus_tqfd.atlas.selfplay import AtlasSelfPlayWorker
from prometheus_tqfd.entropy.trainer import EntropyTrainer
from prometheus_tqfd.entropy.selfplay import EntropySelfPlayWorker
from prometheus_tqfd.evaluation.arena import Arena, NNPlayer
from prometheus_tqfd.evaluation.baselines import RandomPlayer, HeuristicPlayer
from prometheus_tqfd.utils.logging import MetricsLogger

class Supervisor:
    def __init__(self, config):
        self.config = config
        self.stop_event = mp.Event()
        self.pause_event = mp.Event()
        self.oom_event = mp.Event()
        self.training_lock = mp.Lock()

        self.manager = mp.Manager()
        self.shared_values = self.manager.dict({
            'atlas_games': 0,
            'entropy_games': 0,
            'atlas_steps': 0,
            'entropy_steps': 0,
            'atlas_elo': config.elo_initial,
            'entropy_elo': config.elo_initial,
            'status': 'starting'
        })

        self.queues = {
            'atlas_data': mp.Queue(maxsize=100),
            'entropy_data': mp.Queue(maxsize=100),
            'atlas_weights': mp.Queue(maxsize=1),
            'entropy_weights': mp.Queue(maxsize=1),
            'metrics': mp.Queue(maxsize=1000)
        }

        self.cp_mgr = CheckpointManager(config)
        self.recovery_mgr = RecoveryManager(config, self.stop_event, self.oom_event)
        self.logger = MetricsLogger(config.base_dir / config.run_id)

        self.processes = {}

    def run(self):
        print("Supervisor starting...")

        # 1. Start Trainer processes
        self.processes['atlas_trainer'] = mp.Process(
            target=guarded_run,
            args=(self._run_atlas_trainer, 'atlas_trainer', self.recovery_mgr)
        )
        self.processes['entropy_trainer'] = mp.Process(
            target=guarded_run,
            args=(self._run_entropy_trainer, 'entropy_trainer', self.recovery_mgr)
        )

        # 2. Start Self-Play Actors
        for i in range(self.config.num_atlas_actors):
            name = f'atlas_actor_{i}'
            self.processes[name] = mp.Process(
                target=guarded_run,
                args=(self._run_atlas_actor, name, self.recovery_mgr, i)
            )

        for i in range(self.config.num_entropy_actors):
            name = f'entropy_actor_{i}'
            self.processes[name] = mp.Process(
                target=guarded_run,
                args=(self._run_entropy_actor, name, self.recovery_mgr, i)
            )

        # 3. Start Evaluator
        self.processes['evaluator'] = mp.Process(
            target=guarded_run,
            args=(self._run_evaluator, 'evaluator', self.recovery_mgr)
        )

        for p in self.processes.values():
            p.start()

        self.shared_values['status'] = 'running'

        # 4. Supervisor Loop (Watchdog & Checkpointing)
        last_micro_cp = time.time()
        last_light_cp = time.time()
        last_full_cp = time.time()

        try:
            while not self.stop_event.is_set():
                time.sleep(1)

                # Process Metrics
                while not self.queues['metrics'].empty():
                    m = self.queues['metrics'].get()
                    self.logger.log(m)

                # Health Check
                health = self.recovery_mgr.check_system_health()
                if health == "hot":
                    self.pause_event.set()
                    time.sleep(60)
                    self.pause_event.clear()

                # Checkpointing logic
                now = time.time()
                if now - last_micro_cp > 300: # 5m
                    self._save_checkpoint('micro')
                    last_micro_cp = now
                if now - last_light_cp > 900: # 15m
                    self._save_checkpoint('light')
                    last_light_cp = now
                if now - last_full_cp > 3600: # 60m
                    self._save_checkpoint('full')
                    last_full_cp = now

                # Monitor processes
                for name, p in self.processes.items():
                    if not p.is_alive() and not self.stop_event.is_set():
                        print(f"Process {name} died. Restarting...")
                        if 'actor' in name:
                            i = int(name.split('_')[-1])
                            target = self._run_atlas_actor if 'atlas' in name else self._run_entropy_actor
                            new_p = mp.Process(target=guarded_run, args=(target, name, self.recovery_mgr, i))
                        elif name == 'evaluator':
                            new_p = mp.Process(target=guarded_run, args=(self._run_evaluator, name, self.recovery_mgr))
                        elif name == 'atlas_trainer':
                            new_p = mp.Process(target=guarded_run, args=(self._run_atlas_trainer, name, self.recovery_mgr))
                        elif name == 'entropy_trainer':
                            new_p = mp.Process(target=guarded_run, args=(self._run_entropy_trainer, name, self.recovery_mgr))
                        else:
                            continue

                        new_p.start()
                        self.processes[name] = new_p

        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self._save_checkpoint('full')
            for p in self.processes.values():
                p.terminate()
                p.join()

    def _save_checkpoint(self, type):
        print(f"💾 Saving {type} checkpoint...")
        # Collect data from trainers (if they were running and shared their state)
        # In a real implementation, we'd use a shared dict or a request-response pattern.
        # For now, let's assume trainers regularly update a 'latest_weights' in shared_values.

        content = {
            'atlas_model': self.shared_values.get('atlas_weights'),
            'entropy_model': self.shared_values.get('entropy_weights'),
            'atlas_opt': self.shared_values.get('atlas_opt_state'),
            'entropy_opt': self.shared_values.get('entropy_opt_state'),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate()
            },
            'atlas_step': self.shared_values['atlas_steps'],
            'entropy_step': self.shared_values['entropy_steps'],
            'atlas_games': self.shared_values['atlas_games'],
            'entropy_games': self.shared_values['entropy_games'],
        }

        if type == 'full':
            # This is hard via MP queues.
            # Ideally, the trainers would save their buffers to disk and we just move them.
            pass

        self.cp_mgr.save(content, type=type)

    def _run_atlas_trainer(self):
        trainer = AtlasTrainer(self.config, self.queues['atlas_data'], self.queues['atlas_weights'], self.stop_event, self.queues['metrics'], self.shared_values)
        original_train_step = trainer._train_step
        def locked_train_step():
            # ATLAS Priority logic
            acquired = False
            while not acquired and not self.stop_event.is_set():
                if self.training_lock.acquire(timeout=0.1):
                    try:
                        original_train_step()
                        acquired = True
                    finally:
                        self.training_lock.release()
                else:
                    time.sleep(0.01)
        trainer._train_step = locked_train_step
        trainer.run()

    def _run_entropy_trainer(self):
        trainer = EntropyTrainer(self.config, self.queues['entropy_data'], self.queues['entropy_weights'], self.stop_event, self.queues['metrics'], self.shared_values)
        original_train_step = trainer._train_step
        def locked_train_step():
            acquired = False
            while not acquired and not self.stop_event.is_set():
                # ENTROPY might wait if ATLAS wants the lock
                if random.random() < 0.4: # Only try if we pass the 40% chance when competing?
                    # Better: try to acquire, but ATLAS has higher chance or priority
                    pass

                if self.training_lock.acquire(timeout=0.1):
                    try:
                        original_train_step()
                        acquired = True
                    finally:
                        self.training_lock.release()
                else:
                    time.sleep(0.05) # Wait longer than ATLAS
        trainer._train_step = locked_train_step
        trainer.run()

    def _run_atlas_actor(self, worker_id):
        worker = AtlasSelfPlayWorker(self.config, self.queues['atlas_weights'], self.queues['atlas_data'], self.stop_event, worker_id)
        worker.run()

    def _run_entropy_actor(self, worker_id):
        worker = EntropySelfPlayWorker(self.config, self.queues['entropy_weights'], self.queues['entropy_data'], self.stop_event, worker_id)
        worker.run()

    def _run_evaluator(self):
        print("Evaluator started.")
        arena = Arena(self.config, self.queues['metrics'], self.shared_values)

        # We need models to play. Evaluator will load latest weights from shared_values.
        from prometheus_tqfd.atlas.network import AtlasNetwork
        from prometheus_tqfd.entropy.network import EntropyNetwork
        atlas_model = AtlasNetwork(self.config).to(self.config.device)
        entropy_model = EntropyNetwork(self.config).to(self.config.device)

        last_eval_total_games = 0

        while not self.stop_event.is_set():
            current_total_games = self.shared_values['atlas_games'] + self.shared_values['entropy_games']

            if current_total_games - last_eval_total_games >= self.config.eval_interval_games:
                print(f"⚔️ Starting Duels at {current_total_games} games...")

                # Load latest weights
                if self.shared_values.get('atlas_weights'):
                    atlas_model.load_state_dict(self.shared_values['atlas_weights'])
                if self.shared_values.get('entropy_weights'):
                    entropy_model.load_state_dict(self.shared_values['entropy_weights'])

                # Setup players
                p_atlas = NNPlayer(atlas_model, self.config, arena.move_encoder, arena.board_encoder, is_mcts=True)
                p_entropy = NNPlayer(entropy_model, self.config, arena.move_encoder, arena.board_encoder, arena.field_calc, is_mcts=False)
                p_random = RandomPlayer()
                p_heuristic = HeuristicPlayer()

                results = {'atlas_wins': 0, 'entropy_wins': 0, 'draws': 0}

                # 1. Atlas vs Entropy
                for i in range(self.config.eval_games_per_duel):
                    # Alternate colors
                    if i % 2 == 0: res = arena.play_game(p_atlas, p_entropy)
                    else: res = arena.play_game(p_entropy, p_atlas)

                    if res == "1-0":
                        if i % 2 == 0: results['atlas_wins'] += 1
                        else: results['entropy_wins'] += 1
                    elif res == "0-1":
                        if i % 2 == 0: results['entropy_wins'] += 1
                        else: results['atlas_wins'] += 1
                    else:
                        results['draws'] += 1

                arena.update_elo(results)

                # 2. Baselines (optional, for metrics)
                # ... analogous logic for random/heuristic ...

                self.queues['metrics'].put({
                    'type': 'duel',
                    'results': results,
                    'atlas_elo': self.shared_values['atlas_elo'],
                    'entropy_elo': self.shared_values['entropy_elo']
                })

                last_eval_total_games = current_total_games

            time.sleep(60)
