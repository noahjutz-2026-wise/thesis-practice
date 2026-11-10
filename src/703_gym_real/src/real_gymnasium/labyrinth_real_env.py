import pickle
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import zmq
from gymnasium.core import ActType, ObsType
from gymnasium.spaces.box import Box
from numpy.typing import NDArray

TOPIC_KEY_CAMERA = b"cam"  # t,img: tuple[int,cv2.MatLike]
TOPIC_KEY_ACTUATOR = b"actuator"  # action x,y: tuple[int,int]
TOPIC_KEY_VISION_TRACK_CORNERS = (
    b"vision/track_corners"  # imgpts: (8,2) NDArray[np.float64]
)
TOPIC_KEY_VISION_TRACK_BALL = b"vision/track_ball"  # (1,2) NDArray[np.float64]
TOPIC_KEY_VISION_POSE = b"vision/pose"  # rotvec: (3,) NDArray[np.float64]
TOPIC_KEY_MPU = b"mpu"  # (8,) NDArray[np.float64]: (t, ax, ay, az, gx, gy, gz, temp)


class LabyrinthRealEnv(gym.Env):
    def __init__(self, gain: float = 100) -> None:
        super().__init__()
        self.gain = gain
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

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        self.pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps(action * self.gain)])
        obs = self._get_obs()
        reward = np.float64(0)  # TODO
        terminated = False  # TODO
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> NDArray[np.float64]:
        return np.zeros((8,), dtype=np.float64)
