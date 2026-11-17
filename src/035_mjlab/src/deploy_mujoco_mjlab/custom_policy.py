import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.scene.scene import SceneCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer, PolicyProtocol

SCENE_OUT_PATH = Path(__file__).parents[2] / "scene.zip"

scene = SceneCfg()

env_cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    decimation=5,
)


class MyProtocol(PolicyProtocol):
    def __init__(self, env):
        self.shape = (env.num_envs, env.action_manager.total_action_dim)
        self.device = env.device
        self.i = 0

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # return torch.zeros(self.shape, device=self.device)
        self.i += 1
        value = math.sin(self.i * 0.5) * 0.5
        return torch.full(self.shape, value, device=self.device)


def main():
    cfg = load_env_cfg("Mjlab-Cartpole-Balance", play=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    policy = MyProtocol(env)

    NativeMujocoViewer(env, policy).run()


if __name__ == "__main__":
    main()
