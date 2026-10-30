import itertools
from pathlib import Path

import cv2

from mujoco_gymnasium import labyrinth_env

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def main():
    env = labyrinth_env.LabyrinthEnv(str(MUJOCO_XML), 3, render_mode="rgb_array")
    print(env)
    env.reset()

    for i in itertools.count():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)

        if i % 500 == 0:
            frame = env.render()  # RGB, shape (H, W, 3)
            cv2.imshow("sim", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if terminated or truncated:
            obs, info = env.reset()
