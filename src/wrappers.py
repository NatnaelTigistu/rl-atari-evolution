from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Register ALE namespace before any gym.make("ALE/...") call
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError as _ale_err:
    raise ImportError(
        "ale-py is required. Install with: pip install ale-py autorom[accept-rom-license]"
    ) from _ale_err

from config import CFG


class NoopResetEnv(gym.Wrapper):
    """Execute 1–noop_max random no-ops on reset for diverse starting states."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.np_random.integers(1, self.noop_max + 1)):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for games that require it to start (e.g. Pong)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in action_meanings
        self.fire_action = action_meanings.index("FIRE")

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action for `skip` frames; return pixel-wise max of last 2 frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(list(self._obs_buffer), axis=0), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Signal terminated=True on life loss so the value function isn't bootstrapped across lives."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives: int = 0
        self.was_real_done: bool = True

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        current_lives = self.env.unwrapped.ale.lives()
        if current_lives < self.lives and current_lives > 0:
            terminated = True
        self.lives = current_lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # Step with NOOP to get past the life-lost freeze frame
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Resize to 84×84 grayscale. Returns (H, W) 2-D so FrameStack produces (4, 84, 84)."""

    def __init__(self, env: gym.Env,
                 width: int = CFG.FRAME_SIZE[1],
                 height: int = CFG.FRAME_SIZE[0]) -> None:
        super().__init__(env)
        self._width, self._height = width, height
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self._width, self._height), interpolation=cv2.INTER_AREA)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize pixels from [0, 255] → [0.0, 1.0]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=old.shape, dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32) / 255.0


def make_atari_env(
    env_id: str = CFG.ENV_NAME,
    *,
    seed: int = CFG.SEED,
    render_mode: str | None = None,
    episodic_life: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """Build the fully pre-processed Atari env. Final obs shape: (4, 84, 84)."""
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    if clip_reward:
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=CFG.FRAME_STACK)
    env.reset(seed=seed)
    return env


def make_eval_env(
    env_id: str = CFG.ENV_NAME,
    *,
    seed: int = CFG.SEED,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    """Eval variant: episodic_life=False, clip_reward=False for true scores."""
    return make_atari_env(env_id, seed=seed, render_mode=render_mode,
                          episodic_life=False, clip_reward=False)
