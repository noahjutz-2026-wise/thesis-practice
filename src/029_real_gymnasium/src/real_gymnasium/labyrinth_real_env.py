from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import zmq
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.box import Box


class LabyrinthRealEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = Box(-1, 1, (2,))
        self.observation_space = Box(-1, 1, (8,))

        self.pub = zmq.Context.instance().socket(zmq.PUB)
        self.pub.connect("tcp://127.0.0.1:5555")

        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.connect("tcp://127.0.0.1:5556")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs = np.zeros(())
        info = {}
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs = np.zeros(())  # TODO
        reward = 0  # TODO
        terminated = False  # TODO
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
