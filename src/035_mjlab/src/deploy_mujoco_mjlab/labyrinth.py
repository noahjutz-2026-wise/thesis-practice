import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointEffortActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.scene.scene import SceneCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    matrix_from_quat,
    subtract_frame_transforms,
    transform_points,
)
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.debug_visualizer import DebugVisualizer

from deploy_mujoco_mjlab.cartpole import EntityCfg

LABYRINTH_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"
LAYOUT_PATH = (
    Path(__file__).parents[3]
    / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
)


@dataclass(frozen=True)
class PathProgress:
    waypoints: torch.Tensor
    waypoints_interp: torch.Tensor  # Not yet implemented

    @classmethod
    def from_json(cls, layout_path, spacing=0.25, device="cuda"):
        with open(layout_path) as f:
            wp = torch.tensor(json.load(f)["waypoints"], dtype=torch.float32)
        return cls(wp.to(device), wp.to(device))


class VisualizerCommand(CommandTerm):
    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        assert isinstance(cfg, VisualizerCommandCfg)
        self.path_progress = cfg.path_progress

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pts = self.path_progress.waypoints  # (M, 2)
        pts = torch.cat([pts, torch.zeros_like(pts[:, :1])], dim=-1)  # (M, 3)
        p_w, r_w = _board_to_world(self._env, pts)
        mat = matrix_from_quat(r_w)  # (E, M, 3, 3)

        closest = _closest_wp(self._env, pts)

        for e in range(p_w.shape[0]):
            for m in range(mat.shape[1]):
                is_closest = m == closest[e]
                visualizer.add_box(
                    center=p_w[e, m].cpu().numpy(),
                    size=np.ones(3) * 0.005,
                    mat=mat[e, m].cpu().numpy(),
                    color=(1, 0, 0, 1) if not is_closest else (0, 1, 0, 1),
                )


@dataclass(kw_only=True)
class VisualizerCommandCfg(CommandTermCfg):
    path_progress: PathProgress

    def build(self, env: ManagerBasedRlEnv) -> CommandTerm:
        return VisualizerCommand(self, env)


class ProgressDeltaReward(ManagerTermBase):
    def __init__(self, cfg, env) -> None:
        self.pts = cfg.params["pts"].to(env.device)
        self.prev = torch.zeros((env.num_envs,), device=env.device)
        super().__init__(env)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.prev[env_ids] = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        idx = _closest_wp(self._env, self.pts)
        rew = idx - self.prev
        self.prev = idx
        return rew


def _ball_wrt_board(env: ManagerBasedRlEnv) -> torch.Tensor:
    board: Entity = env.scene["board"]
    bi = board.body_names.index("board")
    ball: Entity = env.scene["ball"]

    ball_pos, _ = subtract_frame_transforms(
        board.data.body_link_pos_w[:, bi],
        board.data.body_link_quat_w[:, bi],
        ball.data.root_link_pos_w,
    )
    return ball_pos


def _board_to_world(
    env: ManagerBasedRlEnv, pts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    E = env.num_envs
    board: Entity = env.scene["board"]
    bi = board.body_names.index("board")
    M = pts.shape[0]

    pos = board.data.body_link_pos_w[:, bi]  # (E, 3)
    quat = board.data.body_link_quat_w[:, bi]  # (E, 4)

    # Repeat each pos & quat M times
    pos_f = pos.unsqueeze(1).expand(E, M, 3)  # (E, M, 3)
    pos_f = pos_f.reshape(-1, 3)  # (E*M, 3)
    quat_f = quat.unsqueeze(1).expand(E, M, 4)  # (E, M, 4)
    quat_f = quat_f.reshape(-1, 4)  # (E*M, 4)
    pts_f = pts.unsqueeze(0).expand(E, M, 3).reshape(-1, 3)  # (E*M, 3)

    p_w, r_w = combine_frame_transforms(
        pos_f,
        quat_f,
        pts_f,
    )
    p_w = p_w.view(E, M, 3)  # (E, M, 3)
    r_w = r_w.view(E, M, 4)  # (E, M, 4)
    return p_w, r_w


def _closest(pts: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pts - pt.unsqueeze(1), dim=-1).argmin(dim=-1)


def _closest_wp(env: ManagerBasedRlEnv, pts: torch.Tensor) -> torch.Tensor:
    bp = _ball_wrt_board(env)
    return _closest(pts, bp)


def ball_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball_pos = _ball_wrt_board(env)
    return ball_pos[:, 2] < -0.05


def metric_ball_pos_x(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball_pos = _ball_wrt_board(env)
    return ball_pos[:, 0]


def metric_ball_pos_y(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball_pos = _ball_wrt_board(env)
    return ball_pos[:, 1]


def metric_ball_pos_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball_pos = _ball_wrt_board(env)
    return ball_pos[:, 2]


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


def labyrinth_env_cfg(
    scene_cfg: SceneCfg | None = None, path_progress: PathProgress | None = None
) -> ManagerBasedRlEnvCfg:
    if scene_cfg is None:
        scene_cfg = _get_scene_cfg()

    if path_progress is None:
        path_progress = PathProgress.from_json(LAYOUT_PATH)

    events = {"reset_scene": EventTermCfg(mode="reset", func=reset_scene_to_default)}
    actions: dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="board", actuator_names=("joint_x", "joint_y")
        )
    }
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fallen": TerminationTermCfg(func=ball_fallen),
    }
    metrics = {
        "ball_x": MetricsTermCfg(func=metric_ball_pos_x),
        "ball_y": MetricsTermCfg(func=metric_ball_pos_y),
        "ball_z": MetricsTermCfg(func=metric_ball_pos_z),
        "closest_wp": MetricsTermCfg(
            func=_closest_wp,
            params={"pts": torch.nn.functional.pad(path_progress.waypoints, (0, 1))},
        ),
    }
    rewards = {
        "closest_wp": RewardTermCfg(
            func=ProgressDeltaReward,
            weight=1,
            params={"pts": torch.nn.functional.pad(path_progress.waypoints, (0, 1))},
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "mycmd": VisualizerCommandCfg(
            debug_vis=True,
            resampling_time_range=((0.0, 0.1)),
            path_progress=path_progress,
        )
    }
    env_cfg = ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=5,
        events=events,
        actions=actions,
        terminations=terminations,
        metrics=metrics,
        episode_length_s=10,
        rewards=rewards,
        commands=commands,
    )
    return env_cfg


def labyrinth_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(experiment_name="labyrinth")


def main():
    scene_cfg = _get_scene_cfg()
    env_cfg = labyrinth_env_cfg(scene_cfg)
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
