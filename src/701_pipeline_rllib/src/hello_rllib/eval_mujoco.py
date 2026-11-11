from pathlib import Path

import numpy as np
from mujoco_gymnasium.labyrinth_mujoco_env import LabyrinthMujocoEnv
from mujoco_gymnasium.labyrinth_mujoco_env_wrappers import vec_obs_env
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.spaces.space_utils import unbatch

MUJOCO_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"

CHECKPOINT_PATH = (
    max(
        (Path.home() / "ray_results").glob("**/checkpoint_*"),
        key=lambda p: p.stat().st_mtime,
    )
    / "learner_group/learner/rl_module/default_policy"
)


def main():
    print(CHECKPOINT_PATH)
    env = LabyrinthMujocoEnv(str(MUJOCO_XML), 3, render_mode="human", use_subimg=False)
    env = vec_obs_env(env)
    obs, _ = env.reset()

    ppo_config = PPOConfig()
    dreamerv3_config = DreamerV3Config()

    config = ppo_config
    rl_module = RLModule.from_checkpoint(str(CHECKPOINT_PATH.resolve()))
    env_to_module = config.build_env_to_module_connector(
        env=env, spaces=None, device=None
    )
    module_to_env = config.build_module_to_env_connector(env=env)

    while True:
        episode = SingleAgentEpisode()
        obs, info = env.reset()
        episode.add_env_reset(obs, info)

        terminated = False
        truncated = False
        ret = 0.0
        while not (terminated or truncated):
            shared_data = {}
            batch = env_to_module(
                episodes=[episode],
                rl_module=rl_module,
                batch={},
                explore=False,
                shared_data=shared_data,
            )
            fwd_out = rl_module.forward_inference(batch)
            to_env = module_to_env(
                rl_module=rl_module,
                batch=fwd_out,
                episodes=[episode],
                explore=False,
                shared_data=shared_data,
            )
            action_batch = to_env.pop(Columns.ACTIONS)
            action = unbatch(action_batch)[0]
            # print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            ret += np.float32(reward)
            print(ret)
            episode.add_env_step(
                observation,
                action,
                reward,
                info,
                terminated=terminated,
                truncated=truncated,
            )
