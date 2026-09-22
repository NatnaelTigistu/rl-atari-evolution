"""
config.py
---------
Centralized configuration for the RL-Atari-Evolution project.

All hyper-parameters and paths live here. Import `CFG` anywhere in the
project instead of scattering magic numbers across files.

Usage:
    from config import CFG
    print(CFG.ENV_NAME)       # "ALE/Pong-v5"
    print(CFG.LR)             # 1e-4
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    ENV_NAME: str = "ALE/Pong-v5"

    # Observation preprocessing
    FRAME_SIZE: tuple[int, int] = (84, 84)   # (H, W) after WarpFrame
    FRAME_STACK: int = 4                      # number of stacked frames

    # ------------------------------------------------------------------ #
    # Training hyper-parameters (shared across all algorithms)
    # ------------------------------------------------------------------ #
    GAMMA: float = 0.99        # discount factor
    LR: float = 1e-4           # Adam learning rate
    SEED: int = 42

    # ------------------------------------------------------------------ #
    # A2C / PPO shared
    # ------------------------------------------------------------------ #
    GAE_LAMBDA: float = 0.95   # λ for Generalized Advantage Estimation
    ENTROPY_COEF: float = 0.01 # entropy bonus coefficient
    VALUE_LOSS_COEF: float = 0.5

    # ------------------------------------------------------------------ #
    # PPO-specific
    # ------------------------------------------------------------------ #
    PPO_CLIP_EPS: float = 0.1   # ε for clipped surrogate objective
    PPO_EPOCHS: int = 4         # number of update epochs per rollout
    PPO_MINI_BATCH_SIZE: int = 256
    N_STEPS: int = 2048         # rollout length before each update

    # ------------------------------------------------------------------ #
    # Paths  (all relative to project root; created lazily)
    # ------------------------------------------------------------------ #
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    VIDEO_DIR: str = "videos"

    # Convenience sub-paths
    REINFORCE_CKPT: str = field(default="checkpoints/reinforce")
    A2C_CKPT: str = field(default="checkpoints/a2c")
    PPO_CKPT: str = field(default="checkpoints/ppo")

    REINFORCE_LOG: str = field(default="logs/reinforce_rewards.csv")
    A2C_LOG: str = field(default="logs/a2c_rewards.csv")
    PPO_LOG: str = field(default="logs/ppo_rewards.csv")

    # ------------------------------------------------------------------ #
    # Evaluation / recording
    # ------------------------------------------------------------------ #
    EVAL_EPISODES: int = 10
    RECORD_EPISODES: int = 1

    # ------------------------------------------------------------------ #
    # Display (virtual framebuffer)
    # ------------------------------------------------------------------ #
    DISPLAY_SIZE: tuple[int, int] = (1400, 900)


# ---------------------------------------------------------------------------
# Singleton — import `CFG` everywhere; never instantiate Config yourself.
# ---------------------------------------------------------------------------
CFG = Config()


# ---------------------------------------------------------------------------
# Helper: ensure all output directories exist
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all output directories listed in CFG if they don't exist."""
    dirs = [
        CFG.CHECKPOINT_DIR,
        CFG.LOG_DIR,
        CFG.VIDEO_DIR,
        CFG.REINFORCE_CKPT,
        CFG.A2C_CKPT,
        CFG.PPO_CKPT,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
