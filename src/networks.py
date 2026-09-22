from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import CFG


def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


class NatureCNN(nn.Module):
    """DeepMind Nature CNN: (B, 4, 84, 84) → (B, 512)."""

    _CONV_OUT_SIZE: int = 64 * 7 * 7  # 3136 after three conv layers on 84x84 input

    def __init__(self, in_channels: int = CFG.FRAME_STACK, feature_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _ortho_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)), nn.ReLU(),
            _ortho_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),          nn.ReLU(),
            _ortho_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),          nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            _ortho_init(nn.Linear(self._CONV_OUT_SIZE, feature_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class PolicyNetwork(nn.Module):
    """Policy-only network for REINFORCE. NatureCNN → logits."""

    def __init__(self, action_dim: int = 6, in_channels: int = CFG.FRAME_STACK) -> None:
        super().__init__()
        self.cnn = NatureCNN(in_channels=in_channels)
        self.actor_head = _ortho_init(nn.Linear(512, action_dim), gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.cnn(state))

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        dist = Categorical(logits=self.forward(state))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate_action(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=self.forward(state))
        return dist.log_prob(action), dist.entropy()


class ActorCriticNetwork(nn.Module):
    """Shared CNN backbone with actor and critic heads. Used by A2C and PPO."""

    def __init__(self, action_dim: int = 6, in_channels: int = CFG.FRAME_STACK) -> None:
        super().__init__()
        self.cnn = NatureCNN(in_channels=in_channels)
        self.actor_head  = _ortho_init(nn.Linear(512, action_dim), gain=0.01)
        self.critic_head = _ortho_init(nn.Linear(512, 1), gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.cnn(state)
        return self.actor_head(features), self.critic_head(features)

    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(state)
        return value.squeeze(-1)
