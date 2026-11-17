from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.scene.scene import SceneCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
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
    events = {"reset_scene": reset_scene_to_default}
    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "ball": EntityCfg(
                spec_fn=_get_ball_spec,
                init_state=EntityCfg.InitialStateCfg(pos=(0.01, 0.1, 0.16)),
            ),
            "board": EntityCfg(
                spec_fn=_get_board_spec,
                init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
            ),
        },
        num_envs=2,
    )

    env_cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=5,
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", events=events)
    env = RslRlVecEnvWrapper(env)

    NativeMujocoViewer(
        env,
        lambda obs: torch.zeros(
            env.unwrapped.action_space.shape, device=env.unwrapped.device
        ),
    ).run()


if __name__ == "__main__":
    main()
