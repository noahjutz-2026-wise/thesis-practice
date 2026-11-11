import itertools
from pathlib import Path

import cv2
import numpy as np

from mujoco_gymnasium import labyrinth_env

MUJOCO_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"


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
        # print(cum)

        img = obs["subimg"].reshape(64, 64, 3).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("f", img)
        cv2.waitKey(1)

        if terminated or truncated:
            obs, info = env.reset()
            cum = 0
