"""
src/train.py
------------
Unified training entry point for all three RL algorithms.

Usage (from project root)
--------------------------
# Train REINFORCE for 1 000 episodes, checkpoint every 50:
    python -m src.train --algo reinforce --episodes 1000 --save-freq 50

# Train A2C (once implemented):
    python -m src.train --algo a2c --episodes 3000

# Train PPO (once implemented):
    python -m src.train --algo ppo --episodes 5000

Outputs
-------
  logs/<algo>_rewards.csv   — per-episode reward log
  checkpoints/<algo>/       — model checkpoint .pt files
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ── project root on path so `from config import CFG` always works ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG, ensure_dirs
from src.wrappers import make_atari_env


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RL agent on ALE/Pong-v5"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="reinforce",
        choices=["reinforce", "a2c", "ppo"],
        help="Algorithm to train (default: reinforce)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Total number of training episodes (default: 1000)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        help="Save a checkpoint every N episodes (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CFG.SEED,
        help=f"Random seed (default: {CFG.SEED})",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print status every N episodes (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_agent(algo: str, action_dim: int, device: str):
    """Instantiate the correct agent class for the chosen algorithm."""
    if algo == "reinforce":
        from src.reinforce import REINFORCEAgent
        return REINFORCEAgent(
            action_dim=action_dim,
            lr=CFG.LR,
            gamma=CFG.GAMMA,
            device=device,
        )
    elif algo == "a2c":
        from src.a2c import A2CAgent          # implemented in the next task
        return A2CAgent(
            action_dim=action_dim,
            lr=CFG.LR,
            gamma=CFG.GAMMA,
            device=device,
        )
    elif algo == "ppo":
        from src.ppo import PPOAgent          # implemented in the next task
        return PPOAgent(
            action_dim=action_dim,
            lr=CFG.LR,
            gamma=CFG.GAMMA,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------
class CSVLogger:
    """Append-mode CSV writer that creates the file and header on first write."""

    HEADERS = ["episode", "reward", "length", "loss", "elapsed_s"]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_existed = self.path.exists()

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            if not self._file_existed:
                writer.writeheader()
                self._file_existed = True
            writer.writerow({h: row.get(h, "") for h in self.HEADERS})


# ---------------------------------------------------------------------------
# REINFORCE episode collector
# ---------------------------------------------------------------------------
def run_episode_reinforce(agent, env, device: str) -> tuple[float, int, float]:
    """
    Collect one full episode with REINFORCE and trigger update.

    Returns:
        (total_reward, episode_length, loss)
    """
    obs, _ = env.reset()
    total_reward = 0.0
    ep_len = 0
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_reward(reward)
        total_reward += float(reward)
        ep_len += 1
        done = terminated or truncated

    loss = agent.update()
    return total_reward, ep_len, loss


# ---------------------------------------------------------------------------
# A2C / PPO use a different collect-then-update pattern (stubs here)
# ---------------------------------------------------------------------------
def run_episode_ac(agent, env, device: str) -> tuple[float, int, float]:
    """
    Collect trajectory and update for actor-critic algorithms (A2C / PPO).
    The agent's update() handles its own rollout-length logic.

    Returns:
        (total_reward, episode_length, loss)
    """
    obs, _ = env.reset()
    total_reward = 0.0
    ep_len = 0
    done = False

    while not done:
        obs_t = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0).to(device)

        action, log_prob, value = agent.network.get_action(obs_t)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.store_transition(obs, action, reward, terminated, truncated, log_prob, value)

        obs = next_obs
        total_reward += float(reward)
        ep_len += 1
        done = terminated or truncated

    loss = agent.update(obs, terminated, truncated)
    return total_reward, ep_len, loss


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def ckpt_dir(algo: str) -> Path:
    return Path(CFG.CHECKPOINT_DIR) / algo


def latest_checkpoint(algo: str) -> Path | None:
    """Return the most recently saved checkpoint for `algo`, or None."""
    d = ckpt_dir(algo)
    if not d.exists():
        return None
    pts = sorted(d.glob("ep_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return pts[-1] if pts else None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Device    : {args.device}")
    print(f"  Seed      : {args.seed}")
    print(f"{'='*60}\n")

    # ── Environment ────────────────────────────────────────────────────
    env = make_atari_env(CFG.ENV_NAME, seed=args.seed)
    action_dim = env.action_space.n

    # ── Agent ──────────────────────────────────────────────────────────
    agent = build_agent(args.algo, action_dim, args.device)

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}\n")

    # ── Logger ─────────────────────────────────────────────────────────
    log_path = Path(CFG.LOG_DIR) / f"{args.algo}_rewards.csv"
    logger = CSVLogger(log_path)

    # Running stats (100-episode window)
    reward_window: deque[float] = deque(maxlen=100)

    # ── Dispatch: choose episode runner ────────────────────────────────
    if args.algo == "reinforce":
        run_episode = run_episode_reinforce
    else:
        run_episode = run_episode_ac

    start_time = time.time()

    # ── Training loop ──────────────────────────────────────────────────
    for ep in range(1, args.episodes + 1):
        ep_reward, ep_len, loss = run_episode(agent, env, args.device)
        reward_window.append(ep_reward)
        elapsed = time.time() - start_time

        # ── CSV log ────────────────────────────────────────────────────
        logger.write(
            {
                "episode":   ep,
                "reward":    ep_reward,
                "length":    ep_len,
                "loss":      round(loss, 6),
                "elapsed_s": round(elapsed, 2),
            }
        )

        # ── Console status ─────────────────────────────────────────────
        if ep % args.log_interval == 0:
            avg = np.mean(reward_window)
            print(
                f"Episode: {ep:>5} | "
                f"Score: {ep_reward:>7.1f} | "
                f"Running Avg (100): {avg:>7.2f} | "
                f"Loss: {loss:>10.4f} | "
                f"Time: {elapsed:>8.1f}s"
            )

        # ── Checkpoint ─────────────────────────────────────────────────
        if ep % args.save_freq == 0:
            ckpt_path = ckpt_dir(args.algo) / f"ep_{ep:06d}.pt"
            agent.save(ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Final checkpoint ───────────────────────────────────────────────
    final_path = ckpt_dir(args.algo) / "final.pt"
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Reward log: {log_path}")

    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
