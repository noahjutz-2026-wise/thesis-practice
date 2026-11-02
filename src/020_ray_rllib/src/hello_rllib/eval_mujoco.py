from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"

CHECKPOINT_PATH = Path(__file__).parents[2] / "ckpt"


def main():

    def env(ctx):
        return LabyrinthEnv(str(MUJOCO_XML), 3, render_mode="human")

    # register_env("labyrinth_env", env)

    module = Algorithm.from_checkpoint(str(CHECKPOINT_PATH.resolve()))

    # while True:
