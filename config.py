from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # Environment
    ENV_NAME: str = "ALE/Pong-v5"
    FRAME_SIZE: tuple[int, int] = (84, 84)
    FRAME_STACK: int = 4

    # Shared hyper-parameters
    GAMMA: float = 0.99
    LR: float = 1e-4
    SEED: int = 42

    # A2C / PPO
    GAE_LAMBDA: float = 0.95
    ENTROPY_COEF: float = 0.01
    VALUE_LOSS_COEF: float = 0.5

    # PPO
    PPO_CLIP_EPS: float = 0.1
    PPO_EPOCHS: int = 4
    PPO_MINI_BATCH_SIZE: int = 256
    N_STEPS: int = 2048

    # Output paths
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    VIDEO_DIR: str = "videos"
    REINFORCE_CKPT: str = field(default="checkpoints/reinforce")
    A2C_CKPT: str = field(default="checkpoints/a2c")
    PPO_CKPT: str = field(default="checkpoints/ppo")
    REINFORCE_LOG: str = field(default="logs/reinforce_rewards.csv")
    A2C_LOG: str = field(default="logs/a2c_rewards.csv")
    PPO_LOG: str = field(default="logs/ppo_rewards.csv")

    # Evaluation
    EVAL_EPISODES: int = 10
    RECORD_EPISODES: int = 1
    DISPLAY_SIZE: tuple[int, int] = (1400, 900)


CFG = Config()


def ensure_dirs() -> None:
    for d in [CFG.CHECKPOINT_DIR, CFG.LOG_DIR, CFG.VIDEO_DIR,
              CFG.REINFORCE_CKPT, CFG.A2C_CKPT, CFG.PPO_CKPT]:
        os.makedirs(d, exist_ok=True)
