import itertools

from real_gymnasium.labyrinth_real_env import LabyrinthRealEnv


def main():
    env = LabyrinthRealEnv()

    obs, info = env.reset()

    cum = 0
    for i in itertools.count():
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cum += reward
        print(cum)

        if terminated or truncated:
            obs, info = env.reset()
            cum = 0
