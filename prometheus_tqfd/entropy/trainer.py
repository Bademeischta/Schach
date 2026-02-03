import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import random
import time
from prometheus_tqfd.entropy.rnd import RNDTarget, RNDPredictor
from prometheus_tqfd.entropy.loss import compute_total_entropy_loss

class EntropyReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        for i in range(len(trajectory) - 1):
            step = trajectory[i]
            next_step = trajectory[i+1]
            pair = {
                'field': step['field'],
                'board': step['board'],
                'mask': step['mask'],
                'opp_legal_count': step['opp_legal_count'],
                'energy_captured': step['e_captured'],
                'energy_before': step['energy_before'],
                'energy_next': next_step['energy_before'],
                'energy_diff': next_step['energy_before'] - step['energy_before']
            }
            self.buffer.append(pair)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {
            'fields': torch.stack([item['field'] for item in batch]),
            'boards': torch.stack([item['board'] for item in batch]),
            'masks': torch.stack([item['mask'] for item in batch]),
            'opp_legal_counts': torch.tensor([item['opp_legal_count'] for item in batch]),
            'energy_captured': torch.tensor([item['energy_captured'] for item in batch], dtype=torch.float32),
            'energy_next': torch.tensor([item['energy_next'] for item in batch], dtype=torch.float32),
            'energy_diff': torch.tensor([item['energy_diff'] for item in batch], dtype=torch.float32)
        }

    def __len__(self):
        return len(self.buffer)

class EntropyTrainer:
    def __init__(self, config, data_queue, weights_queue, stop_event, metrics_queue, shared_values=None):
        self.config = config
        self.data_queue = data_queue
        self.weights_queue = weights_queue
        self.stop_event = stop_event
        self.metrics_queue = metrics_queue
        self.shared_values = shared_values

        from prometheus_tqfd.entropy.network import EntropyNetwork
        self.network = EntropyNetwork(config).to(config.device)
        self.optimizer = AdamW(self.network.parameters(), lr=config.entropy_learning_rate)

        self.rnd_target = RNDTarget(1024, config.entropy_rnd_feature_dim).to(config.device)
        self.rnd_predictor = RNDPredictor(1024, config.entropy_rnd_feature_dim).to(config.device)
        self.rnd_optimizer = AdamW(self.rnd_predictor.parameters(), lr=1e-4)

        self.scaler = GradScaler()
        self.replay_buffer = EntropyReplayBuffer(config.entropy_replay_size)
        self.global_step = 0
        self.games_played = 0

    def run(self):
        print("Entropy Trainer started.")
        while not self.stop_event.is_set():
            while not self.data_queue.empty():
                trajectory = self.data_queue.get()
                self.replay_buffer.add(trajectory)
                self.games_played += 1

            if len(self.replay_buffer) >= self.config.min_buffer_before_training:
                self._train_step()

                if self.global_step % self.config.weight_publish_interval == 0:
                    state_dict = self.network.state_dict()
                    self.weights_queue.put(state_dict)
                    if self.shared_values is not None:
                        self.shared_values['entropy_weights'] = state_dict
                        self.shared_values['entropy_opt_state'] = self.optimizer.state_dict()
                        self.shared_values['entropy_steps'] = self.global_step
                        self.shared_values['entropy_games'] = self.games_played
            else:
                time.sleep(1)

    def _train_step(self):
        self.network.train()
        batch = self.replay_buffer.sample(self.config.entropy_batch_size)
        for k in batch:
            batch[k] = batch[k].to(self.config.device)

        self.optimizer.zero_grad()
        self.rnd_optimizer.zero_grad()

        with autocast():
            loss, l_ent, l_cons, l_smooth, l_td, l_novel = compute_total_entropy_loss(
                batch, self.network, self.rnd_target, self.rnd_predictor, self.config
            )

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.step(self.rnd_optimizer)
        self.scaler.update()

        self.global_step += 1

        # Log metrics
        self.metrics_queue.put({
            'type': 'entropy_metrics',
            'step': self.global_step,
            'loss': loss.item(),
            'l_entropy': l_ent,
            'l_conservation': l_cons,
            'l_smoothness': l_smooth,
            'l_td': l_td,
            'l_novelty': l_novel,
            'games': self.games_played,
            'buffer_size': len(self.replay_buffer)
        })
