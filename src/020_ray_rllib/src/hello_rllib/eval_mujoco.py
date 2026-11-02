from pathlib import Path

from mujoco_gymnasium.labyrinth_env import LabyrinthEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.spaces.space_utils import unbatch
from ray.tune.registry import register_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"

CHECKPOINT_PATH = Path(__file__).parents[2] / "ckpt"


def main():
    env = LabyrinthEnv(str(MUJOCO_XML), 3, render_mode="human")
    obs, _ = env.reset()

    config = PPOConfig()
    rl_module = RLModule.from_checkpoint(str(CHECKPOINT_PATH.resolve()))
    env_to_module = config.build_env_to_module_connector(env=env)
    module_to_env = config.build_module_to_env_connector(env=env)

    while True:
        episode = SingleAgentEpisode()
        obs, info = env.reset()
        episode.add_env_reset(obs, info)

        terminated = False
        truncated = False
        while not (terminated or truncated):
            shared_data = {}
            batch = env_to_module(
                episodes=[episode],
                rl_module=rl_module,
                batch={},
                explore=True,
                shared_data=shared_data,
            )
            fwd_out = rl_module.forward_exploration(batch)
            to_env = module_to_env(
                rl_module=rl_module,
                batch=fwd_out,
                episodes=[episode],
                explore=True,
                shared_data=shared_data,
            )
            action_batch = to_env.pop(Columns.ACTIONS)
            action = unbatch(action_batch)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            episode.add_env_step(
                observation,
                action,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
            )
