from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"


def main():
    def env(ctx):
        return LabyrinthEnv(str(MUJOCO_XML), 3, render_mode="human")

    register_env("labyrinth_env", env)

    base_config = PPOConfig()

    config = base_config.environment("labyrinth_env").env_runners(
        num_env_runners=0, num_envs_per_env_runner=1
    )

    ppo = config.build_algo()

    dic = ppo.train()
    print(dic)
