import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.external.env_runner_server_for_external_inference import (
    EnvRunnerServerForExternalInference,
)
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.spaces.space_utils import unbatch


def main():
    config = PPOConfig().environment("CartPole-v1")
    ppo = config.build()
    env = gym.make("CartPole-v1", render_mode="human")

    rl_module: RLModule = ppo.get_module()
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

    exit(0)

    env_runner = SingleAgentEnvRunner(config=config)

    while True:
        obs = env_runner.sample(num_timesteps=1, explore=True)
        rl_module.forward_exploration(batch=obs[0])

    print(rl_module)
