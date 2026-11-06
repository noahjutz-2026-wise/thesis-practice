import itertools
from pathlib import Path

import cv2

from mujoco_gymnasium import labyrinth_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"


def main():
    env = labyrinth_env.LabyrinthEnv(
        str(MUJOCO_XML), 3, render_mode="human", random_starts=False
    )
    print(env)
    env.reset()

    for i in itertools.count():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # if reward != 0:
        #     print(reward)

        if terminated or truncated:
            obs, info = env.reset()
