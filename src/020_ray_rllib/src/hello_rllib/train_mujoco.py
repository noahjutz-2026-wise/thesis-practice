from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from real_gymnasium.labyrinth_real_env import LabyrinthRealEnv

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"
CHECKPOINT_PATH = Path(__file__).parents[2] / "ckpt"

N_ENV_RUNNERS = 28
N_ENVS_PER_RUNNER = 8
ROLLOUT_FRAGMENT_LENGTH = 128


def _restore_checkpoint(*, algorithm: Algorithm, **kwargs):
    try:
        module = algorithm.get_module()
        assert module is not None
        module.restore_from_path(CHECKPOINT_PATH)
    except (ValueError, FileNotFoundError):
        print("unable to restore module")


def _save_checkpoint(*, algorithm: Algorithm, **kwargs):
    print("saving module")
    module = algorithm.get_module()
    assert module is not None
    module.save_to_path(CHECKPOINT_PATH)


def ppo_config():
    return (
        PPOConfig()
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
        .callbacks(
            on_algorithm_init=_restore_checkpoint, on_train_result=_save_checkpoint
        )
    )


def dreamerv3_config():
    return DreamerV3Config().learners(num_learners=0, num_gpus_per_learner=1)


def sim_env():
    name = "labyrinth_game_real"

    def env(ctx):
        return LabyrinthRealEnv()

    register_env(name, env)
    return name


def real_env():
    name = "labyrinth_game_real"

    def env(ctx):
        return LabyrinthEnv(str(MUJOCO_XML), 3)

    register_env(name, env)
    return name


def main():
    tuner = tune.Tuner(
        "PPO",
        run_config=tune.RunConfig(
            callbacks=[WandbLoggerCallback(entity="tjno", project="crs2r_results")],
            stop={"num_env_steps_sampled_lifetime": 2_000_000},
        ),
        param_space=ppo_config,
    )

    tuner.fit()
