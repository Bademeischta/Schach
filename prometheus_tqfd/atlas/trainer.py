import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import random
import time

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, trajectory):
        for step in trajectory:
            self.buffer.append(step)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([s['state'] for s in batch])
        policies = torch.stack([s['policy'] for s in batch])
        values = torch.tensor([s['value'] for s in batch], dtype=torch.float32)
        return states, policies, values

    def __len__(self):
        return len(self.buffer)

class AtlasTrainer:
    def __init__(self, config, data_queue, weights_queue, stop_event, metrics_queue, shared_values=None):
        self.config = config
        self.data_queue = data_queue
        self.weights_queue = weights_queue
        self.stop_event = stop_event
        self.metrics_queue = metrics_queue
        self.shared_values = shared_values

        from prometheus_tqfd.atlas.network import AtlasNetwork
        self.network = AtlasNetwork(config).to(config.device)
        self.optimizer = AdamW(self.network.parameters(), lr=config.atlas_learning_rate, weight_decay=config.atlas_weight_decay)
        self.scaler = GradScaler()

        self.replay_buffer = ReplayBuffer(config.atlas_replay_size)
        self.global_step = 0
        self.games_played = 0

    def run(self):
        print("Atlas Trainer started.")
        while not self.stop_event.is_set():
            # Collect data
            while not self.data_queue.empty():
                trajectory = self.data_queue.get()
                self.replay_buffer.add(trajectory)
                self.games_played += 1

            # Train if enough data
            if len(self.replay_buffer) >= self.config.min_buffer_before_training:
                self._train_step()

                if self.global_step % self.config.weight_publish_interval == 0:
                    state_dict = self.network.state_dict()
                    self.weights_queue.put(state_dict)
                    if self.shared_values is not None:
                        self.shared_values['atlas_weights'] = state_dict
                        self.shared_values['atlas_opt_state'] = self.optimizer.state_dict()
                        self.shared_values['atlas_steps'] = self.global_step
                        self.shared_values['atlas_games'] = self.games_played
            else:
                time.sleep(1)

    def _train_step(self):
        self.network.train()
        states, target_policies, target_values = self.replay_buffer.sample(self.config.atlas_batch_size)
        states, target_policies, target_values = states.to(self.config.device), target_policies.to(self.config.device), target_values.to(self.config.device)

        self.optimizer.zero_grad()
        with autocast():
            policy_logits, value_pred = self.network(states)

            # Policy loss: Cross-entropy
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value_pred.view(-1), target_values)

            total_loss = policy_loss + value_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.global_step += 1

        # Log metrics
        self.metrics_queue.put({
            'type': 'atlas_metrics',
            'step': self.global_step,
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'games': self.games_played,
            'buffer_size': len(self.replay_buffer)
        })
