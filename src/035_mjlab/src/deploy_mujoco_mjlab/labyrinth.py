import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.scene.scene import SceneCfg
from mjlab.viewer import NativeMujocoViewer


def main():
    scene = SceneCfg()

    env_cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=5,
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env)

    NativeMujocoViewer(
        env,
        lambda obs: torch.zeros(
            env.unwrapped.action_space.shape, device=env.unwrapped.device
        ),
    ).run()


if __name__ == "__main__":
    main()
