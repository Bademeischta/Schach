import time
import torch
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
                 weights_queue: Queue, device: str, shared_values: dict):
        self.config = config
        self.data_queue = data_queue
        self.weights_queue = weights_queue
        self.device = device
        self.shared_values = shared_values

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

    def run(self, stop_event: Event, pause_event: Event,
            gpu_lock: Lock, heartbeat_dict: dict, metrics_queue: Queue):
        """Hauptschleife"""
        while not stop_event.is_set():
            heartbeat_dict['entropy_trainer'] = time.time()

            if pause_event.is_set():
                time.sleep(1)
                continue

            self._collect_data()

            if len(self.replay_buffer) >= self.config.min_buffer_before_training:
                with gpu_lock:
                    metrics = self._train_step()
                    metrics_queue.put({
                        'type': 'entropy_train',
                        'step': self.global_step,
            'games': self.shared_values.get('entropy_games', 0),
                        **metrics
                    })

                if self.global_step % self.config.weight_publish_interval == 0:
                    self._publish_weights()
            else:
                time.sleep(1)

    def _collect_data(self):
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
                self.shared_values['entropy_games'] = self.shared_values.get('entropy_games', 0) + 1
        except:
            pass

    def _train_step(self) -> dict:
        self.network.train()
        batch_size = self.config.entropy_batch_size

        states_batch = []
        fields_batch = []
        features_batch = []
        policy_logits_batch = []
        legal_self_batch = []
        legal_opp_batch = []
        energy_next_batch = []
        results_batch = []

        data = self.replay_buffer.sample(batch_size)
        # data is (states, policies, values) from ReplayBuffer.
        # But for entropy we stored tuples in those positions.

        # Wait, the ReplayBuffer I wrote is:
        # def add(self, state, policy, value):
        #    self.buffer.append((state, policy, value))
        # def sample(self, batch_size: int):
        #    batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        #    states, policies, values = zip(*batch)
        #    return (torch.stack(states), torch.stack(policies), torch.tensor(values))

        # In EntropyTrainer._collect_data:
        # self.replay_buffer.add(
        #     (step['board_tensor'], step['field_tensor'], step['features'], step['policy_logits']),
        #     (step['legal_count_self'], step['legal_count_opponent'], step['energy_after']),
        #     step['game_result']
        # )

        # So we need to unpack.
        samples = random.sample(self.replay_buffer.buffer, min(len(self.replay_buffer), batch_size))

        for (s_tup, p_tup, res) in samples:
            states_batch.append(s_tup[0])
            fields_batch.append(s_tup[1])
            features_batch.append(s_tup[2])
            policy_logits_batch.append(s_tup[3])
            legal_self_batch.append(p_tup[0])
            legal_opp_batch.append(p_tup[1])
            energy_next_batch.append(p_tup[2])
            results_batch.append(res)

        states = torch.stack(states_batch).to(self.device)
        fields = torch.stack(fields_batch).to(self.device)
        features = torch.stack(features_batch).to(self.device)
        # policy_logits from buffer are precomputed, but we want the ones from the current model during training?
        # Actually mobility loss uses policy_logits from current forward pass.

        energy_next = torch.tensor(energy_next_batch).float().to(self.device).view(-1, 1)
        legal_self = torch.tensor(legal_self_batch).to(self.device)
        legal_opp = torch.tensor(legal_opp_batch).to(self.device)
        results = torch.tensor(results_batch).float().to(self.device)

        self.optimizer.zero_grad()
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
        self.shared_values['entropy_steps'] = self.global_step

        return {
            'loss': loss.item(),
            **loss_dict
        }

    def _publish_weights(self):
        self.weights_version += 1
        weights = {k: v.cpu() for k, v in self.network.state_dict().items()}
        try:
            while not self.weights_queue.empty():
                self.weights_queue.get_nowait()
        except:
            pass
        self.weights_queue.put((weights, self.weights_version))
        self.shared_values['entropy_weights'] = weights
        self.shared_values['entropy_optimizer'] = self.optimizer.state_dict()
        self.shared_values['entropy_version'] = self.weights_version

import random
