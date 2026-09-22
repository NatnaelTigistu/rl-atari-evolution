"""
src/reinforce.py
----------------
REINFORCE (Monte-Carlo Policy Gradient) agent — Williams 1992.

Algorithm sketch
----------------
For each episode:
  1. Roll out full episode under π_θ, collecting (s_t, a_t, r_t).
  2. Compute discounted returns  G_t = Σ_{k=0}^∞ γ^k r_{t+k}
  3. Normalize G_t  →  (G_t - mean) / (std + ε)
  4. Policy gradient loss = -Σ_t log π_θ(a_t|s_t) · Ĝ_t
  5. Backprop, clip gradients (max norm 0.5), Adam step.

Key correctness notes
---------------------
• Returns are computed from stored raw Python floats — no computational
  graph leaks through the reward buffer.
• log_prob tensors ARE kept on the graph intentionally (they connect the
  update to the current policy parameters).
• select_action() stores log_probs in self._log_probs and rewards are
  appended by the training loop via store_reward(); both are cleared by
  update() at the end of each episode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from src.networks import PolicyNetwork


class REINFORCEAgent:
    """
    Monte-Carlo Policy Gradient agent using the Nature CNN policy network.

    Args:
        action_dim : Number of discrete actions (6 for ALE/Pong-v5).
        lr         : Adam learning rate.
        gamma      : Discount factor γ.
        device     : Torch device string ("cpu" or "cuda").
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

        # Policy network (Nature CNN → actor head)
        self.policy = PolicyNetwork(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Episode rollout buffers
        # ⚠️ log_probs stay as tensors (they must remain on the graph).
        # ⚠️ rewards are raw Python floats (no graph leak).
        self._log_probs: List[torch.Tensor] = []
        self._rewards: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray | torch.Tensor) -> int:
        """
        Sample an action from the current policy π_θ(·|s).

        Stores the log-probability for use in update().

        Args:
            state: Observation array with shape ``(4, 84, 84)`` or a
                   pre-batched tensor ``(1, 4, 84, 84)``.

        Returns:
            Discrete action integer in [0, action_dim).
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(
                np.asarray(state, dtype=np.float32), dtype=torch.float32
            )

        # Ensure batch dimension
        if state.ndim == 3:
            state = state.unsqueeze(0)

        state = state.to(self.device)

        self.policy.eval()   # BN / Dropout off during rollout (future-proof)
        with torch.no_grad():
            # We still need log_prob to have a grad_fn for the update step,
            # so we re-run inside a grad context during update(). Here we
            # just sample the action efficiently.
            action, _ = self.policy.get_action(state)

        # Re-run WITH grad enabled to get a tracked log_prob
        self.policy.train()
        action_tensor, log_prob = self.policy.get_action(state)
        # Override with the same action we already selected (ensures consistency)
        logits = self.policy(state)
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        action_t = torch.tensor([action], device=self.device)
        log_prob = dist.log_prob(action_t)   # shape (1,)

        self._log_probs.append(log_prob)
        return action

    # ------------------------------------------------------------------
    # Reward storage (called by the training loop each step)
    # ------------------------------------------------------------------
    def store_reward(self, reward: float) -> None:
        """Append a step reward to the episode buffer (raw float, no graph)."""
        self._rewards.append(float(reward))

    # ------------------------------------------------------------------
    # Discounted returns
    # ------------------------------------------------------------------
    def compute_returns(self, rewards: List[float] | None = None) -> torch.Tensor:
        """
        Compute normalized discounted returns G_t for the stored episode.

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …

        Normalization: (G - mean(G)) / (std(G) + 1e-8)
        This reduces variance and stabilizes early training.

        Args:
            rewards: Optional external reward list. If None, uses the
                     internal buffer (self._rewards).

        Returns:
            Float32 tensor of shape ``(T,)`` — normalized returns.
        """
        rewards = rewards if rewards is not None else self._rewards

        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize
        mean = returns_t.mean()
        std  = returns_t.std(unbiased=False)  # unbiased=False avoids NaN for T=1
        returns_t = (returns_t - mean) / (std + 1e-8)

        return returns_t

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    def update(self) -> float:
        """
        Perform a single REINFORCE gradient update over the stored episode.

        Loss = -Σ_t log π_θ(a_t|s_t) · Ĝ_t

        Steps:
          1. Compute normalized returns Ĝ from reward buffer.
          2. Stack log_probs into a single tensor.
          3. Compute scalar loss and backprop.
          4. Clip gradients to max-norm 0.5.
          5. Adam step.
          6. Clear rollout buffers.

        Returns:
            Scalar policy loss value (Python float) for logging.
        """
        if not self._rewards:
            return 0.0

        returns = self.compute_returns()                     # (T,)
        log_probs = torch.cat(self._log_probs, dim=0)       # (T,)

        # Sanity check: shapes must match
        assert log_probs.shape == returns.shape, (
            f"log_probs {log_probs.shape} != returns {returns.shape}"
        )

        # Policy gradient loss (negative because we ascend the gradient)
        loss: torch.Tensor = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents catastrophic parameter updates
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)

        self.optimizer.step()

        loss_val = loss.item()
        self._clear_buffers()
        return loss_val

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def _clear_buffers(self) -> None:
        """Reset per-episode rollout buffers."""
        self._log_probs = []
        self._rewards   = []

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        Save policy weights and optimizer state to disk.

        Args:
            path: Target file path (e.g. ``checkpoints/reinforce/ep_100.pt``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict":    self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """
        Load policy weights and optimizer state from disk.

        Args:
            path: Path to a checkpoint saved by ``save()``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[REINFORCE] Loaded checkpoint from {path}")
