from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG, ensure_dirs
from src.wrappers import make_atari_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on ALE/Pong-v5")
    parser.add_argument("--algo",         type=str, default="reinforce",
                        choices=["reinforce", "a2c", "a2c_gae", "ppo"])
    parser.add_argument("--episodes",     type=int, default=1000)
    parser.add_argument("--save-freq",    type=int, default=50)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",       type=str, default=None, metavar="CHECKPOINT_PATH")
    parser.add_argument("--seed",         type=int, default=CFG.SEED)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_agent(algo: str, action_dim: int, device: str):
    if algo == "reinforce":
        from src.reinforce import REINFORCEAgent
        return REINFORCEAgent(action_dim=action_dim, lr=CFG.LR, gamma=CFG.GAMMA, device=device)
    elif algo == "a2c":
        from src.a2c import A2CAgent
        return A2CAgent(action_dim=action_dim, lr=CFG.LR, gamma=CFG.GAMMA, device=device)
    elif algo == "a2c_gae":
        from src.a2c_gae import A2CGAEAgent
        return A2CGAEAgent(action_dim=action_dim, lr=CFG.LR, gamma=CFG.GAMMA,
                           lam=CFG.GAE_LAMBDA, device=device)
    elif algo == "ppo":
        from src.ppo import PPOAgent
        return PPOAgent(action_dim=action_dim, lr=CFG.LR, gamma=CFG.GAMMA, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")


class CSVLogger:
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


def run_episode_reinforce(agent, env, device: str) -> tuple[float, int, float]:
    obs, _ = env.reset()
    total_reward, ep_len, done = 0.0, 0, False
    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_reward(reward)
        total_reward += float(reward)
        ep_len += 1
        done = terminated or truncated
    return total_reward, ep_len, agent.update()


def run_episode_ac(agent, env, device: str) -> tuple[float, int, float]:
    obs, _ = env.reset()
    total_reward, ep_len, done = 0.0, 0, False
    terminated = truncated = False
    while not done:
        obs_t = torch.tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0).to(device)
        action, log_prob, value, entropy = agent.network.get_action(obs_t)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(obs, action, reward, terminated, truncated,
                               log_prob, value, entropy)
        obs = next_obs
        total_reward += float(reward)
        ep_len += 1
        done = terminated or truncated
    return total_reward, ep_len, agent.update(obs, terminated, truncated)


def ckpt_dir(algo: str) -> Path:
    return Path(CFG.CHECKPOINT_DIR) / algo


def latest_checkpoint(algo: str) -> Path | None:
    d = ckpt_dir(algo)
    if not d.exists():
        return None
    pts = sorted(d.glob("ep_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return pts[-1] if pts else None


def train(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Device    : {args.device}")
    print(f"  Seed      : {args.seed}")
    print(f"{'='*60}\n")

    env = make_atari_env(CFG.ENV_NAME, seed=args.seed)
    agent = build_agent(args.algo, env.action_space.n, args.device)

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}\n")

    log_path = Path(CFG.LOG_DIR) / f"{args.algo}_rewards.csv"
    logger = CSVLogger(log_path)
    reward_window: deque[float] = deque(maxlen=100)

    run_episode = run_episode_reinforce if args.algo == "reinforce" else run_episode_ac
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        ep_reward, ep_len, loss = run_episode(agent, env, args.device)
        reward_window.append(ep_reward)
        elapsed = time.time() - start_time

        logger.write({"episode": ep, "reward": ep_reward, "length": ep_len,
                      "loss": round(loss, 6), "elapsed_s": round(elapsed, 2)})

        if ep % args.log_interval == 0:
            avg = np.mean(reward_window)
            print(f"Episode: {ep:>5} | Score: {ep_reward:>7.1f} | "
                  f"Running Avg (100): {avg:>7.2f} | Loss: {loss:>10.4f} | "
                  f"Time: {elapsed:>8.1f}s")

        if ep % args.save_freq == 0:
            ckpt_path = ckpt_dir(args.algo) / f"ep_{ep:06d}.pt"
            agent.save(ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    final_path = ckpt_dir(args.algo) / "final.pt"
    agent.save(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Reward log: {log_path}")
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
