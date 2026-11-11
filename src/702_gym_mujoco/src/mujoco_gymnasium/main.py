import itertools
from pathlib import Path

import cv2
import numpy as np

from mujoco_gymnasium import labyrinth_mujoco_env, labyrinth_mujoco_env_wrappers

MUJOCO_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"


def main():
    env = labyrinth_mujoco_env.LabyrinthMujocoEnv(
        str(MUJOCO_XML), 3, render_mode="human", random_starts=False
    )
    env = labyrinth_mujoco_env_wrappers.vec_obs_env(env)
    print(env.observation_space)
    env.reset()

    cum = 0
    for i in itertools.count():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cum += reward
        # print(cum)

        if terminated or truncated:
            obs, info = env.reset()
            cum = 0
