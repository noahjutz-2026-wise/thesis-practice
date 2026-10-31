from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"


def main():
    def env(ctx):
        return LabyrinthEnv(str(MUJOCO_XML), 3)

    register_env("labyrinth_env", env)

    base_config = PPOConfig()

    config = (
        base_config.environment("labyrinth_env")
        .env_runners(num_env_runners=32, num_envs_per_env_runner=50)
        .learners(num_learners=0, num_gpus_per_learner=1)
    )

    ppo = config.build_algo()

    nan = float("nan")
    for i in range(100):
        r = ppo.train()["env_runners"]
        if i % 5 == 0:
            print(
                f"iter {i:3d} | steps {r.get('num_env_steps_sampled_lifetime', 0):.0f} | "
                f"return {r.get('episode_return_mean', nan):.2f} "
                f"[{r.get('episode_return_min', nan):.1f}, {r.get('episode_return_max', nan):.1f}] | "
                f"len {r.get('episode_len_mean', nan):.1f}"
            )
