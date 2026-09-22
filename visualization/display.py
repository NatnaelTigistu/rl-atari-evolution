"""
visualization/display.py
------------------------
Virtual display management and episode video recording.

Two public functions:
    init_virtual_display()       → start Xvfb if running headless
    record_evaluation_video()    → run agent, save .mp4 with gymnasium RecordVideo
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from config import CFG


# ---------------------------------------------------------------------------
# Protocol — duck-typed Agent interface
# ---------------------------------------------------------------------------
@runtime_checkable
class Agent(Protocol):
    """
    Minimal interface expected from any agent passed to the recorder.

    All three agent classes (REINFORCE, A2C, PPO) must expose
    `select_action(obs)` which returns an integer action.
    """

    def select_action(self, obs: np.ndarray) -> int:
        ...


# ---------------------------------------------------------------------------
# 1. init_virtual_display
# ---------------------------------------------------------------------------
def init_virtual_display(
    size: tuple[int, int] = CFG.DISPLAY_SIZE,
) -> object | None:
    """
    Start a virtual X11 framebuffer (Xvfb) when running in a headless
    environment (e.g. a remote server, Docker container, or Colab notebook).

    Detection heuristic:
        • If the ``DISPLAY`` environment variable is not set, or
        • if it is set but equals an obviously virtual value (":99"),
        → assume headless and launch pyvirtualdisplay.

    Args:
        size: Resolution of the virtual framebuffer (width, height).

    Returns:
        The active ``Display`` object (call ``.stop()`` on it when done),
        or ``None`` if a real display was already available.

    Raises:
        ImportError: If pyvirtualdisplay is not installed.
        RuntimeError: If the display fails to start.
    """
    display_env = os.environ.get("DISPLAY", "")
    is_headless = not display_env or display_env == ":99"

    # Also treat Colab / Jupyter as headless when no real display is set
    in_notebook = "ipykernel" in sys.modules

    if not is_headless and not in_notebook:
        print(f"[display] Real display detected at DISPLAY={display_env!r}. "
              "Skipping virtual display.")
        return None

    try:
        from pyvirtualdisplay import Display  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pyvirtualdisplay is required for headless rendering.\n"
            "Install it with:  pip install pyvirtualdisplay"
        ) from exc

    display = Display(visible=0, size=size)
    display.start()

    # Set DISPLAY so child processes (e.g. ffmpeg) can find the framebuffer
    os.environ.setdefault("DISPLAY", display.env()["DISPLAY"])

    print(
        f"[display] Virtual display started at "
        f"DISPLAY={os.environ['DISPLAY']} "
        f"(size={size[0]}×{size[1]})."
    )
    return display


# ---------------------------------------------------------------------------
# 2. record_evaluation_video
# ---------------------------------------------------------------------------
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
    """
    Run the agent in the environment for ``num_episodes`` and save an .mp4.

    The function imports ``make_eval_env`` from ``src.wrappers`` at call time
    to avoid a circular import (wrappers → config is fine; display → wrappers
    should stay one-directional).

    Args:
        agent:         A trained agent that implements ``select_action(obs)``.
        env_id:        Gymnasium environment ID to record.
        output_path:   Directory where the .mp4 file(s) will be written.
        num_episodes:  Number of full episodes to record.
        seed:          RNG seed for reproducible recordings.
        device:        Torch device for the agent's forward pass.
        video_length:  If set, cap the recording at this many steps.
        name_prefix:   Prefix for the output video file name.

    Returns:
        Path to the directory containing the saved video(s).

    Raises:
        ImportError: If ``gymnasium[other]`` (moviepy / imageio) is missing.
    """
    import gymnasium as gym

    # Import here to avoid module-level circular dependency
    from src.wrappers import make_eval_env  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build a fresh eval env with rgb_array rendering (required by RecordVideo)
    env = make_eval_env(env_id, seed=seed, render_mode="rgb_array")

    # gymnasium.wrappers.RecordVideo wraps the env and writes .mp4 via moviepy
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_path),
        episode_trigger=lambda ep_id: ep_id < num_episodes,
        name_prefix=name_prefix,
        # `video_length` limits the maximum number of recorded steps
        **({"video_length": video_length} if video_length is not None else {}),
    )

    episode_rewards: list[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0

        while not done:
            # Convert LazyFrame / ndarray → float32 tensor
            obs_array = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = agent.select_action(obs_tensor)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        print(f"[record] Episode {ep + 1}/{num_episodes}  reward={ep_reward:.1f}")

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    print(
        f"[record] Saved {num_episodes} episode(s) to '{output_path}'. "
        f"Mean reward: {mean_reward:.1f}"
    )
    return output_path


# ---------------------------------------------------------------------------
# 3. Utility: display context manager
# ---------------------------------------------------------------------------
class VirtualDisplayContext:
    """
    Context manager that starts a virtual display on ``__enter__`` and
    cleanly shuts it down on ``__exit__``.

    Usage::

        with VirtualDisplayContext() as disp:
            record_evaluation_video(agent, ...)
    """

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
