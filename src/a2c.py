from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import CFG
from src.networks import ActorCriticNetwork


class A2CAgent:
    """
    Advantage Actor-Critic with N-step bootstrap returns.

    Collects a full episode of transitions, then computes N-step returns
    backwards from a bootstrapped terminal value before updating.
    """

    def __init__(
        self,
        action_dim: int = 6,
        lr: float = CFG.LR,
        gamma: float = CFG.GAMMA,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Rollout buffers — cleared after every update
        self._log_probs: List[torch.Tensor] = []
        self._values:    List[torch.Tensor] = []
        self._entropies: List[torch.Tensor] = []
        self._rewards:   List[float] = []
        self._dones:     List[float] = []  # 1.0 if terminated/truncated else 0.0

    # ------------------------------------------------------------------
    def store_transition(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> None:
        """Buffer one environment transition. Called from the training loop."""
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(float(reward))
        self._dones.append(float(terminated or truncated))
        # entropy may be None when called from the old stub (graceful fallback)
        if entropy is not None:
            self._entropies.append(entropy)

    # ------------------------------------------------------------------
    def update(
        self,
        next_obs: np.ndarray | torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        Compute N-step returns backwards, derive advantages, update network.

        next_obs    : the observation AFTER the last stored transition.
        terminated  : True if the episode ended due to a game-over.
        truncated   : True if the episode ended due to a time limit.

        Bootstrap rule:
            - terminated → R = 0   (no future value; agent is truly done)
            - truncated  → R = V(next_obs)  (episode cut short; value exists)
        """
        if not self._rewards:
            return 0.0

        T = len(self._rewards)

        # --- Bootstrap value of the state AFTER the rollout ---------------
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(
                np.asarray(next_obs, dtype=np.float32), dtype=torch.float32
            )
        if next_obs.ndim == 3:
            next_obs = next_obs.unsqueeze(0)
        next_obs = next_obs.to(self.device)

        with torch.no_grad():
            next_value = self.network.get_value(next_obs)  # (1,)

        # If truly game-over, don't bootstrap (R starts at 0)
        R: torch.Tensor = torch.zeros_like(next_value) if terminated else next_value

        # --- Compute N-step returns backwards -----------------------------
        returns: List[torch.Tensor] = []
        for r, d in zip(reversed(self._rewards), reversed(self._dones)):
            R = r + self.gamma * R * (1.0 - d)
            returns.insert(0, R)

        returns_t   = torch.stack(returns).squeeze(-1).to(self.device)      # (T,)
        log_probs_t = torch.cat(self._log_probs, dim=0)                     # (T,)
        values_t    = torch.cat(self._values, dim=0)                        # (T,)

        # --- Advantages ---------------------------------------------------
        advantages = returns_t - values_t.detach()
        # Normalize for stable CNN training (prevents exploding policy updates)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # --- Losses -------------------------------------------------------
        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t.detach())

        if self._entropies:
            entropy_bonus = torch.cat(self._entropies, dim=0).mean()
        else:
            # Fallback: recompute entropy from stored log_probs (approximation)
            entropy_bonus = -log_probs_t.mean()

        total_loss = actor_loss + CFG.VALUE_LOSS_COEF * critic_loss - CFG.ENTROPY_COEF * entropy_bonus

        # --- Update -------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        loss_val = total_loss.item()
        self._clear_buffers()
        return loss_val

    # ------------------------------------------------------------------
    def _clear_buffers(self) -> None:
        self._log_probs  = []
        self._values     = []
        self._entropies  = []
        self._rewards    = []
        self._dones      = []

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network_state_dict":   self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[A2C] Loaded checkpoint from {path}")
