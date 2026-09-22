from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from src.networks import PolicyNetwork


class REINFORCEAgent:
    """Monte-Carlo Policy Gradient agent (Williams 1992)."""

    def __init__(
        self,
        action_dim: int = 6,
        lr: float = CFG.LR,
        gamma: float = CFG.GAMMA,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy = PolicyNetwork(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self._log_probs: List[torch.Tensor] = []  # kept on graph for backprop
        self._rewards: List[float] = []            # raw floats, no graph leak

    def select_action(self, state: np.ndarray | torch.Tensor) -> int:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.asarray(state, dtype=np.float32), dtype=torch.float32)
        if state.ndim == 3:
            state = state.unsqueeze(0)  # (1, 4, 84, 84)
        state = state.to(self.device)
        self.policy.train()
        action, log_prob = self.policy.get_action(state)
        self._log_probs.append(log_prob)
        return action

    def store_reward(self, reward: float) -> None:
        self._rewards.append(float(reward))

    def compute_returns(self, rewards: List[float] | None = None) -> torch.Tensor:
        rewards = rewards if rewards is not None else self._rewards
        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
        return returns_t

    def update(self) -> float:
        if not self._rewards:
            return 0.0
        returns = self.compute_returns()
        log_probs = torch.cat(self._log_probs, dim=0)
        assert log_probs.shape == returns.shape
        loss: torch.Tensor = -(log_probs * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        loss_val = loss.item()
        self._clear_buffers()
        return loss_val

    def _clear_buffers(self) -> None:
        self._log_probs = []
        self._rewards = []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[REINFORCE] Loaded checkpoint from {path}")
