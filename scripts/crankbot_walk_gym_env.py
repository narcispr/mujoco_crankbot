from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

try:
    from crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig
except ImportError:
    from .crankbot_walk_env import CrankBotWalkEnv, CrankBotWalkEnvConfig


class CrankBotWalkGymEnv(CrankBotWalkEnv, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        cfg: CrankBotWalkEnvConfig | None = None,
        device: str = "cpu",
        render_mode: str | None = None,
    ) -> None:
        self.render_mode = render_mode
        self._gym_initializing = True
        CrankBotWalkEnv.__init__(self, replace(cfg or CrankBotWalkEnvConfig(), num_envs=1), device=device)
        self._gym_initializing = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.num_privileged_obs,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if getattr(self, "_gym_initializing", False):
            return CrankBotWalkEnv.reset(self)

        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        CrankBotWalkEnv.reset(self)
        return self._privileged_obs(), self._info({})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device).reshape(1, self.num_actions)
        _, reward, done, extras = CrankBotWalkEnv.step(self, action_tensor)

        truncated = bool(extras["time_outs"][0].item())
        terminated = bool(done[0].item() and not truncated)
        return self._privileged_obs(), float(reward[0].item()), terminated, truncated, self._info(extras)

    def render(self) -> None:
        return None

    def _privileged_obs(self) -> np.ndarray:
        return self.get_privileged_observations()[0].detach().cpu().numpy().astype(np.float32, copy=True)

    def _info(self, extras: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "goal_position": self.goal_pos_world[0].astype(np.float32, copy=True),
            "goal_bearing": float(self.goal_bearing[0]),
            "goal_range": float(self.goal_range[0]),
            "goal_reached": bool(self.goal_reached[0]),
            "base_velocity_body": self.base_velocity_body[0].astype(np.float32, copy=True),
        }
        if "log" in extras:
            info["log"] = {
                key: float(value.detach().cpu().item()) if torch.is_tensor(value) else value
                for key, value in extras["log"].items()
            }
        return info
