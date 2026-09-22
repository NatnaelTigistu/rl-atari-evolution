"""
src/networks.py
---------------
Neural network architectures for all three RL algorithms.

Architecture (DeepMind Nature CNN — Mnih et al. 2015)
-----------------------------------------------------
Input shape : (B, 4, 84, 84)  — 4 stacked 84×84 grayscale frames

CNN backbone
    Conv1 : 4  → 32 ch, kernel 8×8, stride 4  → ReLU
    Conv2 : 32 → 64 ch, kernel 4×4, stride 2  → ReLU
    Conv3 : 64 → 64 ch, kernel 3×3, stride 1  → ReLU
    Flatten: 64 × 7 × 7 = 3136
    FC1   : 3136 → 512                         → ReLU

Heads
    PolicyNetwork (REINFORCE)
        actor head : 512 → action_dim  (raw logits)

    ActorCriticNetwork (A2C / PPO)
        actor head  : 512 → action_dim  (raw logits)
        critic head : 512 → 1           (state value V(s))

Weight initialisation
    Conv / FC layers : orthogonal, gain = √2
    Actor output     : orthogonal, gain = 0.01  (small init → near-uniform
                       distribution at the start → better exploration)
    Critic output    : orthogonal, gain = 1.0

⚠️  Logit / Softmax pitfall
    torch.distributions.Categorical accepts either raw ``logits=`` or
    normalised ``probs=``.  We always use ``logits=`` and never manually
    apply softmax before passing to the distribution.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import CFG


# ---------------------------------------------------------------------------
# Orthogonal initialisation helper
# ---------------------------------------------------------------------------
def _ortho_init(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    """Apply orthogonal initialisation to ``module``'s weight and zero bias."""
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)
    return module


# ---------------------------------------------------------------------------
# Shared CNN backbone
# ---------------------------------------------------------------------------
class NatureCNN(nn.Module):
    """
    DeepMind Nature CNN feature extractor.

    Accepts a batch of stacked frames with shape ``(B, 4, 84, 84)`` and
    produces a 512-dimensional feature vector for each element in the batch.

    The output of this module is a **512-dim ReLU-activated** tensor — it is
    intentionally not a raw linear output so that actor / critic heads can be
    stacked on top without needing their own activation.

    Args:
        in_channels: Number of input channels (= FRAME_STACK = 4).
        feature_dim: Output dimensionality of FC1 (default: 512).
    """

    # Spatial size after the three conv layers for an 84×84 input
    _CONV_OUT_SIZE: int = 64 * 7 * 7  # = 3136

    def __init__(
        self,
        in_channels: int = CFG.FRAME_STACK,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # Conv1: (B, 4, 84, 84) → (B, 32, 20, 20)
            _ortho_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            # Conv2: (B, 32, 20, 20) → (B, 64, 9, 9)
            _ortho_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            # Conv3: (B, 64, 9, 9)  → (B, 64, 7, 7)
            _ortho_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            # Flatten + FC1: 3136 → 512
            nn.Flatten(),
            _ortho_init(nn.Linear(self._CONV_OUT_SIZE, feature_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float32 tensor of shape ``(B, 4, 84, 84)``, pixel values in
               ``[0, 1]`` (already scaled by ScaledFloatFrame wrapper).

        Returns:
            Feature tensor of shape ``(B, 512)``.
        """
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# 1. PolicyNetwork — for REINFORCE
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Policy network for the REINFORCE algorithm.

    Architecture: NatureCNN → Linear(512, action_dim)

    The actor head uses a small orthogonal gain (0.01) so that the initial
    action distribution is close to uniform, enabling broad early exploration.

    Args:
        action_dim:  Number of discrete actions (6 for ALE/Pong-v5).
        in_channels: Number of stacked input frames (default: 4).
    """

    def __init__(
        self,
        action_dim: int = 6,
        in_channels: int = CFG.FRAME_STACK,
    ) -> None:
        super().__init__()

        self.cnn = NatureCNN(in_channels=in_channels)

        # Actor head — small gain → near-uniform logits at initialisation
        self.actor_head = _ortho_init(
            nn.Linear(512, action_dim), gain=0.01
        )

    # ------------------------------------------------------------------
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute raw action logits.

        Args:
            state: Float32 tensor ``(B, 4, 84, 84)``.

        Returns:
            Raw logits tensor ``(B, action_dim)``.
        """
        features = self.cnn(state)
        return self.actor_head(features)

    # ------------------------------------------------------------------
    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample an action from the policy and return its log-probability.

        ⚠️  Logits are passed directly to ``Categorical(logits=...)``; we
        never call softmax manually before constructing the distribution.

        Args:
            state: Float32 tensor ``(1, 4, 84, 84)`` (single observation).

        Returns:
            action   : int — the sampled discrete action.
            log_prob : scalar ``torch.Tensor`` — log π(a|s).
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()               # shape (B,)
        log_prob = dist.log_prob(action)     # shape (B,)
        return action.item(), log_prob

    # ------------------------------------------------------------------
    def evaluate_action(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log-probability and entropy of given (state, action) pairs.

        Used during the REINFORCE policy-gradient update to re-compute
        log π(a|s) for the entire collected trajectory in a single batched
        forward pass.

        Args:
            state:  Float32 tensor ``(B, 4, 84, 84)``.
            action: Long tensor   ``(B,)`` — actions taken during rollout.

        Returns:
            log_prob : tensor ``(B,)`` — log π(a|s) for each transition.
            entropy  : tensor ``(B,)`` — H[π(·|s)] for each state
                       (used to compute an optional entropy bonus).
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


# ---------------------------------------------------------------------------
# 2. ActorCriticNetwork — shared backbone for A2C and PPO
# ---------------------------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    """
    Shared CNN backbone with separate actor and critic linear heads.

    Used by both A2C and PPO.  A single forward pass through ``NatureCNN``
    produces the 512-dim feature vector that feeds **both** heads, avoiding
    the computational redundancy of two independent CNNs.

    Architecture:
        NatureCNN (shared)
            ├── actor_head  : Linear(512, action_dim)  → raw logits
            └── critic_head : Linear(512, 1)            → V(s)

    Weight init:
        • Conv / FC1 (inside NatureCNN) : orthogonal, gain = √2
        • actor_head                    : orthogonal, gain = 0.01
        • critic_head                   : orthogonal, gain = 1.0

    Args:
        action_dim:  Number of discrete actions (6 for ALE/Pong-v5).
        in_channels: Number of stacked input frames (default: 4).
    """

    def __init__(
        self,
        action_dim: int = 6,
        in_channels: int = CFG.FRAME_STACK,
    ) -> None:
        super().__init__()

        self.cnn = NatureCNN(in_channels=in_channels)

        # Actor head — small gain for uniform init
        self.actor_head = _ortho_init(
            nn.Linear(512, action_dim), gain=0.01
        )
        # Critic head — standard gain
        self.critic_head = _ortho_init(
            nn.Linear(512, 1), gain=1.0
        )

    # ------------------------------------------------------------------
    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single shared forward pass.

        Args:
            state: Float32 tensor ``(B, 4, 84, 84)``.

        Returns:
            logits : tensor ``(B, action_dim)`` — raw actor logits.
            value  : tensor ``(B, 1)``          — critic state value V(s).
        """
        features = self.cnn(state)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    # ------------------------------------------------------------------
    def get_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action and return action, log_prob, and V(s).

        ⚠️  Uses ``logits=`` (not ``probs=``) to construct ``Categorical``.

        Args:
            state: Float32 tensor ``(1, 4, 84, 84)`` (single observation).

        Returns:
            action   : int            — sampled discrete action.
            log_prob : tensor ``(B,)``— log π(a|s).
            value    : tensor ``(B,)``— V(s) from the critic head.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze(-1)

    # ------------------------------------------------------------------
    def evaluate_actions(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of (state, action) pairs during the update step.

        Called by A2C and PPO to re-compute log-probs, entropy, and values
        for all transitions in the collected mini-batch.

        ⚠️  Tensor detachment is the responsibility of the *caller* (the
        agent's ``update()`` method) — this method performs a clean forward
        pass and should never see stale graph references.

        Args:
            state:  Float32 tensor ``(B, 4, 84, 84)``.
            action: Long tensor   ``(B,)`` — actions taken during rollout.

        Returns:
            log_prob : tensor ``(B,)``  — log π(a|s).
            entropy  : tensor ``(B,)``  — H[π(·|s)].
            value    : tensor ``(B,)``  — V(s) from the critic.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)

    # ------------------------------------------------------------------
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Return only V(s) — used for GAE bootstrapping at the end of a
        truncated rollout without sampling an action.

        Args:
            state: Float32 tensor ``(1, 4, 84, 84)``.

        Returns:
            value: tensor ``(B,)`` — V(s).
        """
        _, value = self.forward(state)
        return value.squeeze(-1)
