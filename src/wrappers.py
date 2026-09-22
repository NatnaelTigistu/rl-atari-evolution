"""
src/wrappers.py
---------------
Standard Atari pre-processing wrappers compatible with gymnasium >= 0.29.

All wrappers handle the 5-tuple step API:
    obs, reward, terminated, truncated, info = env.step(action)

Wrapper chain applied by `make_atari_env()`:
    Raw ALE/Pong-v5
        → NoopResetEnv          (random 1-30 no-ops at reset)
        → MaxAndSkipEnv         (frame-skip=4, pixel-wise max over last 2)
        → EpisodicLifeEnv       (treat each life loss as episode end)
        → FireResetEnv          (press FIRE to launch ball)
        → WarpFrame             (84×84 grayscale)
        → ScaledFloatFrame      (pixels ÷ 255 → [0, 1])
        → FrameStack(4)         (stack last 4 frames → shape (4, 84, 84))
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import CFG


# ---------------------------------------------------------------------------
# 1. NoopResetEnv
# ---------------------------------------------------------------------------
class NoopResetEnv(gym.Wrapper):
    """
    Sample a random number of no-op actions (1 to `noop_max`) on reset.

    This breaks the correlation between episodes that would otherwise always
    start in the same ALE state, giving the agent more diverse starting
    positions.

    Args:
        env:       The environment to wrap.
        noop_max:  Maximum number of no-ops to sample (default: 30).
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # action 0 is NOOP in all ALE games

        assert env.unwrapped.get_action_meanings()[0] == "NOOP", (
            "Action 0 must be NOOP for this wrapper to work correctly."
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)

        n_noops = self.np_random.integers(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()

        return obs, info


# ---------------------------------------------------------------------------
# 2. FireResetEnv
# ---------------------------------------------------------------------------
class FireResetEnv(gym.Wrapper):
    """
    Press FIRE (action 1) immediately after reset for games that require it.

    Pong will not serve the ball until FIRE is pressed; without this wrapper
    the agent would be stuck watching a motionless screen.

    Only applied when the environment's action list contains "FIRE".
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in action_meanings, (
            "FireResetEnv should only wrap environments that have a FIRE action."
        )
        self.fire_action = action_meanings.index("FIRE")

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs, _, terminated, truncated, _ = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(seed=seed, options=options)
        return obs, info


# ---------------------------------------------------------------------------
# 3. MaxAndSkipEnv
# ---------------------------------------------------------------------------
class MaxAndSkipEnv(gym.Wrapper):
    """
    Repeat every action for `skip` frames, accumulate reward, and return the
    pixel-wise maximum of the last two frames.

    The max-pooling over adjacent frames removes the flickering that the ALE
    emulator introduces by alternating sprite rendering across frames.

    Args:
        env:  The environment to wrap.
        skip: Number of frames to repeat each action (default: 4).
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        # Buffer for the two most recent raw observations
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if terminated or truncated:
                break

        # Pixel-wise max to remove flicker artefacts
        max_frame = np.max(np.stack(list(self._obs_buffer), axis=0), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


# ---------------------------------------------------------------------------
# 4. EpisodicLifeEnv
# ---------------------------------------------------------------------------
class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat each life-loss as a terminal signal during training.

    When the agent loses a life the wrapper sets `terminated = True` so the
    value function is not bootstrapped across life boundaries.  The episode
    itself only truly ends when all lives are exhausted; at that point the
    environment is fully reset.

    This is a common trick that gives the agent a much tighter credit-
    assignment signal in games like Pong (3 lives per game).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives: int = 0
        self.was_real_done: bool = True  # track whether the full game ended

    def step(
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        # Check remaining lives via the ALE lives counter
        current_lives = self.env.unwrapped.ale.lives()
        life_lost = current_lives < self.lives and current_lives > 0
        self.lives = current_lives

        # Treat a life-loss as a local terminal signal
        if life_lost:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[Any, dict]:
        # Only do a full reset when the game is actually over
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # Step with NOOP to advance past the "life-lost" freeze frame
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)

        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# ---------------------------------------------------------------------------
# 5. WarpFrame
# ---------------------------------------------------------------------------
class WarpFrame(gym.ObservationWrapper):
    """
    Convert RGB frames to 84×84 grayscale using OpenCV.

    Output observation shape: (84, 84, 1)  — the channel dimension is kept
    so that FrameStack can concatenate along axis=-1 or axis=0.

    Args:
        env:    The environment to wrap.
        width:  Target width  (default: CFG.FRAME_SIZE[1] = 84).
        height: Target height (default: CFG.FRAME_SIZE[0] = 84).
    """

    def __init__(
        self,
        env: gym.Env,
        width: int = CFG.FRAME_SIZE[1],
        height: int = CFG.FRAME_SIZE[0],
    ) -> None:
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        return resized[:, :, np.newaxis]  # (H, W, 1)


# ---------------------------------------------------------------------------
# 6. ScaledFloatFrame
# ---------------------------------------------------------------------------
class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize pixel values from [0, 255] → [0.0, 1.0] (float32).

    This prevents the large initial gradients that raw uint8 inputs would
    cause in the first layers of the CNN.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Update observation space dtype and bounds
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# 7. Factory — make_atari_env
# ---------------------------------------------------------------------------
def make_atari_env(
    env_id: str = CFG.ENV_NAME,
    *,
    seed: int = CFG.SEED,
    render_mode: str | None = None,
    episodic_life: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """
    Build the fully pre-processed Atari environment.

    Wrapper chain:
        gymnasium.make
            → NoopResetEnv   (random 1-30 no-ops at reset)
            → MaxAndSkipEnv  (frame-skip=4, pixel-wise max)
            → EpisodicLifeEnv  (optional; off for evaluation)
            → FireResetEnv   (press FIRE if game requires it)
            → WarpFrame      (84×84 grayscale, shape (84,84,1))
            → ScaledFloatFrame ([0,1] float32)
            → RecordEpisodeStatistics (tracks ep_return & ep_len in info)
            → FrameStack(4)  (final shape (4, 84, 84))

    Args:
        env_id:        Gymnasium environment ID.
        seed:          RNG seed forwarded to env.reset().
        render_mode:   "rgb_array" for recording, None for training.
        episodic_life: Apply EpisodicLifeEnv (recommended during training
                       only; disable for evaluation).
        clip_reward:   Clip rewards to {-1, 0, +1} (recommended during
                       training; disable for evaluation to see true scores).

    Returns:
        A fully wrapped gymnasium environment ready for training.
    """
    env = gym.make(env_id, render_mode=render_mode)

    # -- Core wrappers (order matters) --
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    # FireResetEnv: only apply if the game has a FIRE action
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)
    env = ScaledFloatFrame(env)

    # Clip rewards to {-1, 0, +1} for training stability
    if clip_reward:
        env = gym.wrappers.TransformReward(env, lambda r: np.sign(r))

    # Track episode statistics (ep_return, ep_len) automatically in info dict
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Stack 4 most recent frames; final obs shape: (4, 84, 84)
    env = gym.wrappers.FrameStack(env, num_stack=CFG.FRAME_STACK)

    env.reset(seed=seed)
    return env


def make_eval_env(
    env_id: str = CFG.ENV_NAME,
    *,
    seed: int = CFG.SEED,
    render_mode: str | None = "rgb_array",
) -> gym.Env:
    """
    Convenience wrapper for evaluation / recording runs.

    Disables episodic-life and reward clipping so true game scores are
    reported. Enables rgb_array render mode for video capture.
    """
    return make_atari_env(
        env_id,
        seed=seed,
        render_mode=render_mode,
        episodic_life=False,
        clip_reward=False,
    )
