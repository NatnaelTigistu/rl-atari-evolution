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


class PPOAgent:
    """
    Proximal Policy Optimization (Schulman et al. 2017).

    Collects a full episode rollout, computes GAE advantages, then runs
    K_EPOCHS of clipped surrogate updates over random MINI_BATCH_SIZE
    mini-batches — reusing each transition multiple times safely via the
    clipped ratio constraint.
    """

    K_EPOCHS:       int   = CFG.PPO_EPOCHS           # 4
    MINI_BATCH_SIZE: int  = CFG.PPO_MINI_BATCH_SIZE  # 256 (overridden to 64 below for sanity)
    CLIP_EPS:       float = CFG.PPO_CLIP_EPS          # 0.1

    # Use a smaller mini-batch for this episodic variant so there are always
    # enough samples even in short episodes.
    _MB: int = 64

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

        # Rollout buffers — cleared after every update
        self._states:    List[np.ndarray]    = []   # raw obs for policy re-eval
        self._actions:   List[int]           = []
        self._old_log_probs: List[torch.Tensor] = []  # log π_old(a|s) — detached
        self._values:    List[torch.Tensor]  = []   # V(s_t) — detached
        self._rewards:   List[float]         = []
        self._dones:     List[float]         = []

    # ------------------------------------------------------------------
    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor | None = None,  # accepted but not buffered
    ) -> None:
        """Buffer one step. obs is the raw numpy observation (for re-evaluation)."""
        self._states.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(action)
        self._old_log_probs.append(log_prob.detach())   # must be detached — old policy
        self._values.append(value.detach())
        self._rewards.append(float(reward))
        self._dones.append(float(terminated or truncated))

    # ------------------------------------------------------------------
    def update(
        self,
        next_obs: np.ndarray | torch.Tensor,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        Compute GAE over the stored rollout, then run K_EPOCHS of clipped
        PPO updates over random mini-batches.
        """
        if not self._rewards:
            return 0.0

        T = len(self._rewards)

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

        if terminated:
            next_value = torch.zeros_like(next_value)

        # --- Stack rollout tensors ------------------------------------
        states_t    = torch.tensor(
            np.stack(self._states), dtype=torch.float32, device=self.device
        )                                                       # (T, 4, 84, 84)
        actions_t   = torch.tensor(
            self._actions, dtype=torch.long, device=self.device
        )                                                       # (T,)
        old_lp_t    = torch.cat(self._old_log_probs, dim=0)    # (T,) — old policy, detached
        values_t    = torch.cat(self._values, dim=0)            # (T,) — detached
        rewards_t   = torch.tensor(self._rewards, dtype=torch.float32, device=self.device)
        dones_t     = torch.tensor(self._dones,   dtype=torch.float32, device=self.device)

        # --- GAE: advantages (normalized) and returns (critic targets) -
        advantages, returns = compute_gae(
            rewards_t, values_t, dones_t, next_value,
            gamma=self.gamma, lam=self.lam,
        )
        advantages = advantages.detach()  # must not flow gradients into the ratio
        returns    = returns.detach()

        # --- K epochs of mini-batch PPO updates -----------------------
        indices = torch.randperm(T, device=self.device)
        mb_size = min(self._MB, T)       # never exceed rollout length
        total_loss_sum = 0.0
        num_updates = 0

        for _ in range(self.K_EPOCHS):
            # Shuffle each epoch independently
            perm = torch.randperm(T, device=self.device)

            for start in range(0, T, mb_size):
                mb_idx = perm[start : start + mb_size]
                if len(mb_idx) < 2:
                    continue   # skip lone-sample mini-batches (std would be 0)

                mb_states  = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_lp  = old_lp_t[mb_idx]
                mb_adv     = advantages[mb_idx]
                mb_ret     = returns[mb_idx]

                # Normalize advantages per mini-batch for stable updates
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Re-evaluate with CURRENT policy
                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    mb_states, mb_actions
                )

                # Probability ratio r_t(θ) = π_θ(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - mb_old_lp)

                # Clipped surrogate objective
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic MSE loss
                critic_loss = F.mse_loss(new_values, mb_ret)

                # Entropy bonus (encourages exploration)
                entropy_bonus = entropy.mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_sum += loss.item()
                num_updates += 1

        self._clear_buffers()
        return total_loss_sum / max(num_updates, 1)

    # ------------------------------------------------------------------
    def _clear_buffers(self) -> None:
        self._states      = []
        self._actions     = []
        self._old_log_probs = []
        self._values      = []
        self._rewards     = []
        self._dones       = []

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
        print(f"[PPO] Loaded checkpoint from {path}")
