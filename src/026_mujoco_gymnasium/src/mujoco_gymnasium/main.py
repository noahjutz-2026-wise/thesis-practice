import itertools
from pathlib import Path

import cv2

from mujoco_gymnasium import labyrinth_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"


def main():
    env = labyrinth_env.LabyrinthEnv(
        str(MUJOCO_XML), 3, render_mode="human", random_starts=True
    )
    print(env)
    env.reset()

    cum = 0
    for i in itertools.count():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cum += reward
        print(cum)

        cv2.imshow("f", obs["image"])
        cv2.waitKey(1)

        if terminated or truncated:
            obs, info = env.reset()
