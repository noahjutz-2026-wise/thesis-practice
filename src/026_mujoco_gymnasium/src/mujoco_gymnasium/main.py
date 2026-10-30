from pathlib import Path

from mujoco_gymnasium import labyrinth_env

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def main():
    env = labyrinth_env.LabyrinthEnv(str(MUJOCO_XML), 3)
    print(env)
    env.reset()

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
