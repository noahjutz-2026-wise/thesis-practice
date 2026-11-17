from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.scene.scene import SceneCfg
from mjlab.viewer import NativeMujocoViewer

from deploy_mujoco_mjlab.cartpole import EntityCfg

LABYRINTH_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"


def _get_board_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(LABYRINTH_XML))
    spec.delete(spec.body("ball"))
    for sensor in list(spec.sensors):
        if "ball" in sensor.name:
            spec.delete(sensor)
    return spec


def _get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(LABYRINTH_XML))
    spec.delete(spec.body("board"))
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def main():
    scene = SceneCfg(
        entities={
            "ball": EntityCfg(spec_fn=_get_ball_spec),
            "board": EntityCfg(spec_fn=_get_board_spec),
        }
    )

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
