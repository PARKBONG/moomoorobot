import gymnasium as gym
import torch
import numpy as np
from typing import Tuple


# ============================================================================
# Environment Wrapper
# ============================================================================
class CartPole:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self) -> Tuple[torch.Tensor, np.ndarray]:
        state, _ = self.env.reset()
        return torch.tensor(state, dtype=torch.float32), state

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, np.ndarray]:
        next_state, reward, done, _, _ = self.env.step(action)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        return next_state_tensor, reward, done, next_state

    def close(self):
        self.env.close()
