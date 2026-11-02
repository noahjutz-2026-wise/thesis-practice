import itertools
from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"
CHECKPOINT_PATH = Path(__file__).parents[2] / "ckpt"

N_ENV_RUNNERS = 28
N_ENVS_PER_RUNNER = 8
ROLLOUT_FRAGMENT_LENGTH = 128


def main():
    def env(ctx):
        return LabyrinthEnv(str(MUJOCO_XML), 3)

    register_env("labyrinth_env", env)

    base_config = PPOConfig()

    config = (
        base_config.environment("labyrinth_env")
        .env_runners(
            num_env_runners=N_ENV_RUNNERS,
            num_envs_per_env_runner=N_ENVS_PER_RUNNER,
            rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        )
        .learners(num_learners=0, num_gpus_per_learner=1)
        .training(
            train_batch_size_per_learner=N_ENV_RUNNERS
            * N_ENVS_PER_RUNNER
            * ROLLOUT_FRAGMENT_LENGTH
        )
    )

    ppo = config.build_algo()

    try:
        module = ppo.get_module()
        assert module is not None
        module.restore_from_path(CHECKPOINT_PATH)
    except (ValueError, FileNotFoundError):
        print("unable to restore checkpoint")

    nan = float("nan")
    for i in itertools.count():
        r = ppo.train()["env_runners"]
        if i % 5 == 0:
            print(
                f"iter {i:3d} | steps {r.get('num_env_steps_sampled_lifetime', 0):.0f} | "
                f"return {r.get('episode_return_mean', nan):.2f} "
                f"[{r.get('episode_return_min', nan):.1f}, {r.get('episode_return_max', nan):.1f}] | "
                f"len {r.get('episode_len_mean', nan):.1f}"
            )
        if i % 10 == 0:
            module = ppo.get_module()
            assert module is not None
            module.save_to_path(CHECKPOINT_PATH)
