import random
import torch
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, policies, values = zip(*batch)

        return (
            torch.stack(states),
            torch.stack(policies),
            torch.tensor(values, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def get_data(self):
        return list(self.buffer)

    def set_data(self, data):
        self.buffer.extend(data)
