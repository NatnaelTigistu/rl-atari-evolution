from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from config import CFG


@runtime_checkable
class Agent(Protocol):
    def select_action(self, obs: np.ndarray) -> int: ...


def init_virtual_display(size: tuple[int, int] = CFG.DISPLAY_SIZE) -> object | None:
    """Start Xvfb if running headless (Colab / server). Returns the Display or None."""
    display_env = os.environ.get("DISPLAY", "")
    is_headless = not display_env or display_env == ":99"
    in_notebook  = "ipykernel" in sys.modules

    if not is_headless and not in_notebook:
        print(f"[display] Real display at DISPLAY={display_env!r}, skipping virtual display.")
        return None

    try:
        from pyvirtualdisplay import Display  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Install pyvirtualdisplay: pip install pyvirtualdisplay") from exc

    display = Display(visible=0, size=size)
    display.start()
    os.environ.setdefault("DISPLAY", display.env()["DISPLAY"])
    print(f"[display] Virtual display started at DISPLAY={os.environ['DISPLAY']} ({size[0]}×{size[1]})")
    return display


def record_evaluation_video(
    agent: Agent,
    env_id: str = CFG.ENV_NAME,
    output_path: str | Path = CFG.VIDEO_DIR,
    num_episodes: int = CFG.RECORD_EPISODES,
    *,
    seed: int = CFG.SEED,
    device: str | torch.device = "cpu",
    video_length: int | None = None,
    name_prefix: str = "eval",
) -> Path:
    """Run agent for num_episodes, save .mp4 via RecordVideo, return output dir."""
    import gymnasium as gym
    from src.wrappers import make_eval_env  # imported here to avoid circular import

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    env = make_eval_env(env_id, seed=seed, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_path),
        episode_trigger=lambda ep_id: ep_id < num_episodes,
        name_prefix=name_prefix,
        **({} if video_length is None else {"video_length": video_length}),
    )

    episode_rewards: list[float] = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ep_reward = False, 0.0
        while not done:
            obs_t = torch.tensor(
                np.asarray(obs, dtype=np.float32), dtype=torch.float32
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.select_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        print(f"[record] Episode {ep + 1}/{num_episodes}  reward={ep_reward:.1f}")

    env.close()
    print(f"[record] Saved to '{output_path}'. Mean reward: {np.mean(episode_rewards):.1f}")
    return output_path


class VirtualDisplayContext:
    """Context manager wrapper around init_virtual_display."""

    def __init__(self, size: tuple[int, int] = CFG.DISPLAY_SIZE) -> None:
        self._size = size
        self._display: object | None = None

    def __enter__(self) -> "VirtualDisplayContext":
        self._display = init_virtual_display(self._size)
        return self

    def __exit__(self, *_: object) -> None:
        if self._display is not None and hasattr(self._display, "stop"):
            self._display.stop()  # type: ignore[union-attr]
            print("[display] Virtual display stopped.")
