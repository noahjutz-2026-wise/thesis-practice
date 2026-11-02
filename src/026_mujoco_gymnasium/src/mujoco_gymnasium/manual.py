from pathlib import Path

import cv2
import numpy as np

import mujoco_gymnasium.labyrinth_env as labyrinth_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"
STEP = 0.02
KP = 4.0
KD = 1.2


def main():
    env = labyrinth_env.LabyrinthEnv(str(MUJOCO_XML), 3, render_mode="rgb_array")
    env.reset()
    target = np.zeros(2)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == 81:
            target[1] -= STEP
        elif key == 83:
            target[1] += STEP
        elif key == 82:
            target[0] += STEP
        elif key == 84:
            target[0] -= STEP
        target = np.clip(target, -0.15, 0.15)
        angles = np.array(
            [
                env.data.sensor("sensor_board_rot_x").data[0],
                env.data.sensor("sensor_board_rot_y").data[0],
            ]
        )
        vel = env.data.qvel[:2]
        action = np.clip(KP * (target - angles) - KD * vel, -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if reward != 0:
            print(reward)
        frame = env.render()
        cv2.imshow("manual", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if terminated or truncated:
            env.reset()
            target = np.zeros(2)


if __name__ == "__main__":
    main()
