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
from src.math_utils import compute_gae


class A2CGAEAgent:
    """A2C with Generalized Advantage Estimation (GAE-λ)."""

    def __init__(
        self,
        action_dim: int = 6,
        lr: float = CFG.LR,
        gamma: float = CFG.GAMMA,
        lam: float = CFG.GAE_LAMBDA,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self._log_probs: List[torch.Tensor] = []
        self._values:    List[torch.Tensor] = []
        self._entropies: List[torch.Tensor] = []
        self._rewards:   List[float] = []
        self._dones:     List[float] = []

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
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(float(reward))
        self._dones.append(float(terminated or truncated))
        if entropy is not None:
            self._entropies.append(entropy)

    # ------------------------------------------------------------------
    def update(
        self,
        next_obs: np.ndarray | torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if not self._rewards:
            return 0.0

        # --- Bootstrap next value -------------------------------------
        if not isinstance(next_obs, torch.Tensor):
            next_obs = torch.tensor(
                np.asarray(next_obs, dtype=np.float32), dtype=torch.float32
            )
        if next_obs.ndim == 3:
            next_obs = next_obs.unsqueeze(0)
        next_obs = next_obs.to(self.device)

        with torch.no_grad():
            next_value = self.network.get_value(next_obs)  # (1,)

        # Truly game-over → no future value to bootstrap
        if terminated:
            next_value = torch.zeros_like(next_value)

        # --- Stack rollout buffers ------------------------------------
        rewards_t   = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        dones_t     = torch.tensor(self._dones,   dtype=torch.float32, device=self.device)
        values_t    = torch.cat(self._values, dim=0)    # (T,)
        log_probs_t = torch.cat(self._log_probs, dim=0) # (T,)

        # --- GAE ------------------------------------------------------
        advantages, returns = compute_gae(
            rewards_t, values_t.detach(), dones_t, next_value,
            gamma=self.gamma, lam=self.lam,
        )

        # --- Losses ---------------------------------------------------
        actor_loss    = -(log_probs_t * advantages).mean()
        critic_loss   = F.mse_loss(values_t, returns.detach())
        entropy_bonus = (
            torch.cat(self._entropies, dim=0).mean()
            if self._entropies
            else -log_probs_t.mean()  # fallback approximation
        )

        total_loss = (
            actor_loss
            + CFG.VALUE_LOSS_COEF * critic_loss
            - CFG.ENTROPY_COEF * entropy_bonus
        )

        # --- Update ---------------------------------------------------
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
        print(f"[A2C-GAE] Loaded checkpoint from {path}")
