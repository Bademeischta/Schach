import time
import torch
import torch.nn.functional as F
from multiprocessing import Queue, Event, Lock
from prometheus_tqfd.config import PrometheusConfig
from prometheus_tqfd.atlas.network import AtlasNetwork
from prometheus_tqfd.utils.replay_buffer import ReplayBuffer

class AtlasTrainer:
    """
    Trainiert das ATLAS-Netzwerk mit Daten aus Self-Play.
    """

    def __init__(self, config: PrometheusConfig, data_queue: Queue,
                 weights_queue: Queue, device: str, shared_values: dict):
        self.config = config
        self.data_queue = data_queue
        self.weights_queue = weights_queue
        self.device = device
        self.shared_values = shared_values

        self.network = AtlasNetwork(config).to(device)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.atlas_learning_rate,
            weight_decay=config.atlas_weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

        self.replay_buffer = ReplayBuffer(config.atlas_replay_size)
        self.global_step = 0
        self.weights_version = 0

    def run(self, stop_event: Event, pause_event: Event,
            gpu_lock: Lock, heartbeat_dict: dict, metrics_queue: Queue):
        """Hauptschleife des Trainers"""
        while not stop_event.is_set():
            # Heartbeat
            heartbeat_dict['atlas_trainer'] = time.time()

            # Pause prüfen
            if pause_event.is_set():
                time.sleep(1)
                continue

            # Daten aus Queue holen
            self._collect_data()

            # Training wenn genug Daten
            if len(self.replay_buffer) >= self.config.min_buffer_before_training:
                with gpu_lock:
                    metrics = self._train_step()
                    metrics_queue.put({
                        'type': 'atlas_train',
                        'step': self.global_step,
                        **metrics
                    })

                # Gewichte publishen
                if self.global_step % self.config.weight_publish_interval == 0:
                    self._publish_weights()
            else:
                time.sleep(1) # Wait for data

    def _collect_data(self):
        """Trajektorien aus Queue in Replay Buffer"""
        try:
            while not self.data_queue.empty():
                trajectory = self.data_queue.get_nowait()
                for state, policy, value in trajectory:
                    self.replay_buffer.add(state, policy, value)
        except:
            pass

    def _train_step(self) -> dict:
        """Ein Trainingsschritt"""
        self.network.train()
        states, policies, values = self.replay_buffer.sample(self.config.atlas_batch_size)
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)

        self.optimizer.zero_grad()

        # Cast to device-appropriate autocast
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'

        with torch.autocast(device_type=device_type):
            policy_logits, value_pred = self.network(states)

            # Policy Loss: Cross-Entropy
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(policies * log_probs, dim=1).mean()

            # Value Loss: MSE
            value_loss = F.mse_loss(value_pred.squeeze(-1), values)

            # Total Loss
            loss = policy_loss + value_loss

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        self.global_step += 1
        self.shared_values['atlas_steps'] = self.global_step

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'grad_norm': grad_norm.item() if grad_norm is not None else 0.0
        }

    def _publish_weights(self):
        """Gewichte an Self-Play-Worker senden"""
        self.weights_version += 1
        weights = {k: v.cpu() for k, v in self.network.state_dict().items()}

        # Queue leeren und neue Gewichte einfügen
        try:
            while not self.weights_queue.empty():
                self.weights_queue.get_nowait()
        except:
            pass

        self.weights_queue.put((weights, self.weights_version))

        # Für Checkpoint
        self.shared_values['atlas_weights'] = weights
        self.shared_values['atlas_optimizer'] = self.optimizer.state_dict()
        self.shared_values['atlas_version'] = self.weights_version
