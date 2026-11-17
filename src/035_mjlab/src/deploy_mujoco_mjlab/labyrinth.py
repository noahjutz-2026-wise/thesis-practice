from pathlib import Path

import mujoco
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
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


def _get_scene_cfg() -> SceneCfg:
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
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        XmlActuatorCfg(target_names_expr=("joint_x", "joint_y")),
                    )
                ),
            ),
        },
        num_envs=2,
    )
    return scene


def rl_env_cfg(scene_cfg: SceneCfg) -> ManagerBasedRlEnvCfg:
    events = {"reset_scene": EventTermCfg(mode="reset", func=reset_scene_to_default)}
    actions: dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="board", actuator_names=("joint_x", "joint_y")
        )
    }
    env_cfg = ManagerBasedRlEnvCfg(
        scene=scene_cfg, decimation=5, events=events, actions=actions
    )
    return env_cfg


def main():
    scene_cfg = _get_scene_cfg()
    env_cfg = rl_env_cfg(scene_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env)

    NativeMujocoViewer(
        env,
        lambda obs: torch.ones(
            env.unwrapped.action_space.shape, device=env.unwrapped.device
        ),
    ).run()


if __name__ == "__main__":
    main()
