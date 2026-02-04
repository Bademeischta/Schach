import time
import torch
import random
from multiprocessing import Queue, Event, Lock
from typing import Dict
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.entropy.network import EntropyNetworkV2
from prometheus_tqfd.entropy.loss import EntropyV2Loss
from prometheus_tqfd.utils.replay_buffer import ReplayBuffer

class EntropyTrainer:
    """
    Trainiert das ENTROPY-Netzwerk.
    """

    def __init__(self, config: PrometheusConfig, data_queue: Queue,
                 weights_queue: Queue, device: str, shared_values: dict, shared_lock: Lock):
        self.config = config
        self.data_queue = data_queue
        self.weights_queue = weights_queue
        self.device = device
        self.shared_values = shared_values
        self.shared_lock = shared_lock

        self.network = EntropyNetworkV2(config).to(device)
        self.loss_fn = EntropyV2Loss(config, device)

        self.optimizer = torch.optim.AdamW(
            list(self.network.parameters()) + list(self.loss_fn.rnd_predictor.parameters()),
            lr=config.entropy_learning_rate
        )
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

        self.replay_buffer = ReplayBuffer(config.entropy_replay_size)
        self.global_step = 0
        self.weights_version = 0
        self.metrics_accumulator = []

    def run(self, stop_event: Event, pause_event: Event,
            gpu_lock: Lock, heartbeat_dict: dict, metrics_queue: Queue):
        """Hauptschleife"""
        while not stop_event.is_set():
            heartbeat_dict['entropy_trainer'] = time.time()

            if pause_event.is_set():
                time.sleep(0.5)
                continue

            self._collect_data()

            if len(self.replay_buffer) >= self.config.min_buffer_before_training:
                with gpu_lock:
                    metrics = self._train_step()
                    self.metrics_accumulator.append(metrics)

                if self.global_step % 50 == 0:
                    self._process_periodic_tasks(metrics_queue)

                if self.global_step % self.config.weight_publish_interval == 0:
                    self._publish_weights()
            else:
                time.sleep(1)

    def _collect_data(self):
        new_games = 0
        try:
            while not self.data_queue.empty():
                trajectory = self.data_queue.get_nowait()
                for step in trajectory:
                    # We need to store (board, fields, features, policy_logits, legal_count_self, energy_after, legal_count_opponent, game_result)
                    self.replay_buffer.add(
                        (step['board_tensor'], step['field_tensor'], step['features'], step['policy_logits']),
                        (step['legal_count_self'], step['legal_count_opponent'], step['energy_after']),
                        step['game_result']
                    )
                new_games += 1

            if new_games > 0:
                with self.shared_lock:
                    self.shared_values['entropy_games'] = self.shared_values.get('entropy_games', 0) + new_games
        except:
            pass

    def _train_step(self) -> dict:
        self.network.train()
        batch_size = self.config.entropy_batch_size

        samples = random.sample(self.replay_buffer.buffer, min(len(self.replay_buffer), batch_size))

        states = torch.stack([s[0][0] for s in samples]).to(self.device, non_blocking=True)
        fields = torch.stack([s[0][1] for s in samples]).to(self.device, non_blocking=True)
        features = torch.stack([s[0][2] for s in samples]).to(self.device, non_blocking=True)
        energy_next = torch.tensor([s[1][2] for s in samples]).float().to(self.device, non_blocking=True).view(-1, 1)
        legal_self = torch.tensor([s[1][0] for s in samples]).to(self.device, non_blocking=True)
        legal_opp = torch.tensor([s[1][1] for s in samples]).to(self.device, non_blocking=True)
        results = torch.tensor([s[2] for s in samples]).float().to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        with torch.autocast(device_type=device_type):
            policy_logits, energy = self.network(states, fields)
            batch_data = {
                'states': states,
                'policy_logits': policy_logits,
                'energy': energy,
                'energy_next': energy_next,
                'legal_counts_self': legal_self,
                'legal_counts_opponent': legal_opp,
                'features': features
            }
            loss, loss_dict = self.loss_fn.compute(batch_data, results)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        self.global_step += 1
        return {
            'loss': loss.item(),
            **loss_dict
        }

    def _process_periodic_tasks(self, metrics_queue: Queue):
        with self.shared_lock:
            self.shared_values['entropy_steps'] = self.global_step
            games = self.shared_values.get('entropy_games', 0)

        if self.metrics_accumulator:
            avg_metrics = {
                k: sum(m[k] for m in self.metrics_accumulator) / len(self.metrics_accumulator)
                for k in self.metrics_accumulator[0]
            }
            metrics_queue.put({
                'type': 'entropy_train',
                'step': self.global_step,
                'games': games,
                **avg_metrics
            })
            self.metrics_accumulator = []

    def _publish_weights(self):
        self.weights_version += 1
        weights = {k: v.cpu() for k, v in self.network.state_dict().items()}
        try:
            while not self.weights_queue.empty():
                self.weights_queue.get_nowait()
        except:
            pass
        self.weights_queue.put((weights, self.weights_version))
        with self.shared_lock:
            self.shared_values['entropy_weights'] = weights
            self.shared_values['entropy_optimizer'] = self.optimizer.state_dict()
            self.shared_values['entropy_version'] = self.weights_version
