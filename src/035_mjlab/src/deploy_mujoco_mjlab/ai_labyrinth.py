"""Labyrinth (ball-on-tilting-board) environment for mjlab.

Mirrors the gymnasium env in src/702_gym_mujoco but keeps only the core
behavior: tilt the board with two motors to roll the ball along the waypoint
path. Reference (not copy) of the XML from src/704_backend_mujoco.
"""

import json
from pathlib import Path

import mujoco
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default, resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

LABYRINTH_XML = Path(__file__).parents[3] / "704_backend_mujoco/export.xml"
LAYOUT_PATH = (
    Path(__file__).parents[3]
    / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
)

WINDOW = 0.01  # normalized progress +/- used for reward subpath
BALL_START_X = 0.01  # default ball body pos from the XML
BALL_START_Y = 0.1
BALL_START_Z = 0.01
PENALTY = 0.0001
FALL_Z = -0.05  # ball world z below this counts as fallen off the board

# --- waypoint path (loaded once at import) ----------------------------------

with open(LAYOUT_PATH, "r") as f:
    _WAYPOINTS = torch.tensor(json.load(f)["waypoints"], dtype=torch.float32)

_SEG = _WAYPOINTS[1:] - _WAYPOINTS[:-1]
_SEG_LEN = _SEG.norm(dim=-1)
_CUM = torch.cat([torch.zeros(1), torch.cumsum(_SEG_LEN, dim=0)]) / _SEG_LEN.sum()

# Dense resampling of the path for lookahead targets.
_DENSE_N = 1000
_u = torch.linspace(0.0, 1.0, _DENSE_N)
_seg_idx = torch.clamp(torch.searchsorted(_CUM, _u, right=True) - 1, 0, len(_SEG) - 1)
_t = (_u - _CUM[_seg_idx]) / (_CUM[_seg_idx + 1] - _CUM[_seg_idx]).clamp_min(1e-9)
_DENSE = _WAYPOINTS[_seg_idx] + _t.unsqueeze(-1) * _SEG[_seg_idx]

_BOARD_CFG = SceneEntityCfg("board", joint_names=("joint_x", "joint_y"))


def _name_geoms(spec: mujoco.MjSpec, prefix: str) -> None:
    """Name unnamed geoms by shape type so CollisionCfg patterns can match."""
    names = {
        mujoco.mjtGeom.mjGEOM_MESH: "mesh",
        mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
        mujoco.mjtGeom.mjGEOM_PLANE: "plane",
    }
    for i, geom in enumerate(spec.geoms):
        if not geom.name:
            kind = names.get(geom.type, f"type{int(geom.type)}")
            geom.name = f"{prefix}_{kind}_{i}"


def _get_board_spec() -> mujoco.MjSpec:
    """Board-only spec: remove the detached ball body and its sensors."""
    spec = mujoco.MjSpec.from_file(str(LABYRINTH_XML))
    spec.delete(spec.body("ball"))
    for sensor in list(spec.sensors):
        if "ball" in sensor.name:
            spec.delete(sensor)
    _name_geoms(spec, "board")
    return spec


def _get_ball_spec() -> mujoco.MjSpec:
    """Ball-only spec: a floating body on its own (mjlab requires each
    detached floating body to be its own scene entity)."""
    spec = mujoco.MjSpec.from_file(str(LABYRINTH_XML))
    spec.delete(spec.body("board"))
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    _name_geoms(spec, "ball")
    return spec


# Collision masks: board geoms (4/2) and ball geom (2/4) collide with each
# other but not with the terrain planes (1/1). This keeps per-env contact
# counts small — otherwise the 560-mesh board vs ground plane floods the
# contact buffers and OOMs the GPU at high num_envs.
_BOARD_COLLISIONS = (
    CollisionCfg(
        geom_names_expr=("board_mesh_.*",),
        contype=4,
        conaffinity=2,
        condim=3,
        priority=0,
    ),
)
_BALL_COLLISIONS = (
    CollisionCfg(
        geom_names_expr=("ball_sphere_.*",),
        contype=2,
        conaffinity=4,
        condim=3,
        priority=0,
    ),
)


_BOARD_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=("joint_x", "joint_y")),),
)


def _get_board_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_board_spec,
        articulation=_BOARD_ARTICULATION,
        collisions=_BOARD_COLLISIONS,
        init_state=EntityCfg.InitialStateCfg(),
    )


def _get_ball_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_ball_spec,
        collisions=_BALL_COLLISIONS,
        init_state=EntityCfg.InitialStateCfg(),
    )


# --- helpers ----------------------------------------------------------------


def _ball_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    return ball.data.body_link_pos_w[:, 0, :2]


def _project_onto_path(env: ManagerBasedRlEnv, bp: torch.Tensor):
    """Nearest projection of ball xy onto the path, restricted to a window
    around the current normalized progress. Returns (progress, distance)."""
    dev = bp.device
    a = _WAYPOINTS[:-1].to(dev)
    ab = _SEG.to(dev)
    cum = _CUM.to(dev)
    lensq = (ab**2).sum(-1).clamp_min(1e-12)

    pa = bp.unsqueeze(1) - a.unsqueeze(0)  # (n, M, 2)
    t = ((pa * ab.unsqueeze(0)).sum(-1) / lensq.unsqueeze(0)).clamp(0.0, 1.0)
    proj = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
    d = (bp.unsqueeze(1) - proj).norm(dim=-1)  # (n, M)
    arc = cum[:-1].unsqueeze(0) + t * (cum[1:] - cum[:-1]).unsqueeze(0)

    p = env._labyrinth_progress.to(dev).unsqueeze(1)
    valid = (arc >= (p - WINDOW).clamp(0, 1)) & (arc <= (p + WINDOW).clamp(0, 1))
    d = torch.where(valid, d, torch.full_like(d, float("inf")))
    idx = d.argmin(dim=1)
    arange = torch.arange(bp.shape[0], device=dev)
    newp = arc[arange, idx]
    dist = d[arange, idx]
    newp = torch.where(torch.isinf(dist), p.squeeze(1), newp)
    return newp, dist


def _ensure_progress(env: ManagerBasedRlEnv) -> None:
    if not hasattr(env, "_labyrinth_progress"):
        env._labyrinth_progress = torch.zeros(env.num_envs, device=env.device)


# --- mdp terms ---------------------------------------------------------------


def labyrinth_progress_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Progress along the waypoint path minus a small per-step penalty."""
    _ensure_progress(env)
    newp, _ = _project_onto_path(env, _ball_xy(env))
    reward = newp - env._labyrinth_progress - PENALTY
    env._labyrinth_progress = newp
    return reward


def ball_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    z = ball.data.body_link_pos_w[:, 0, 2]
    return z < FALL_Z


def goal_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_progress(env)
    return env._labyrinth_progress >= 1.0 - 1e-6


def ball_target_vecs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Vectors from the ball to lookahead targets on the path (4 x 2)."""
    _ensure_progress(env)
    dev = env.device
    bp = _ball_xy(env)
    dense = _DENSE.to(dev)
    idx = env._labyrinth_progress.unsqueeze(1) + torch.tensor(
        [0.01, 0.02, 0.04, 0.08], device=dev
    ).unsqueeze(0)
    idx = (idx.clamp(0.0, 1.0) * (_DENSE_N - 1)).long()
    targets = dense[idx]  # (n, 4, 2)
    return (targets - bp.unsqueeze(1)).reshape(bp.shape[0], -1)


def ball_state_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    pos = ball.data.body_link_pos_w[:, 0, :2]
    vel = ball.data.body_link_lin_vel_w[:, 0, :2]
    return torch.cat([pos, vel], dim=-1)


def reset_ball_at_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Reset the ball to the XML default start pose and progress to 0."""
    _ensure_progress(env)
    env_ids = resolve_env_ids(env, env_ids)
    ball: Entity = env.scene["ball"]
    n = len(env_ids)
    dev = env.device

    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = BALL_START_X
    pose[:, 1] = BALL_START_Y
    pose[:, 2] = BALL_START_Z
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
    env._labyrinth_progress[env_ids] = 0.0


# --- env cfg -----------------------------------------------------------------


def _make_labyrinth_cfg() -> ManagerBasedRlEnvCfg:
    tilt_terms = {
        "tilt_pos": ObservationTermCfg(
            func=joint_pos_rel, params={"asset_cfg": _BOARD_CFG}
        ),
        "tilt_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _BOARD_CFG}
        ),
        "ball_state": ObservationTermCfg(func=ball_state_obs),
        "target_vecs": ObservationTermCfg(func=ball_target_vecs),
    }
    observations = {
        "actor": ObservationGroupCfg(tilt_terms),
        "critic": ObservationGroupCfg({**tilt_terms}),
    }
    actions = {
        "effort": JointEffortActionCfg(
            entity_name="board",
            actuator_names=("joint_x", "joint_y"),
            scale=1.0,
        ),
    }
    rewards = {
        "progress": RewardTermCfg(func=labyrinth_progress_reward, weight=1.0),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "fallen": TerminationTermCfg(func=ball_fallen),
        "goal": TerminationTermCfg(func=goal_reached),
    }
    events = {
        "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
        "reset_ball": EventTermCfg(
            func=reset_ball_at_start,
            mode="reset",
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"board": _get_board_cfg(), "ball": _get_ball_cfg()},
            num_envs=1,
            env_spacing=4.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.002),
            # Contact buffers; per-env contacts are small now that only
            # ball<->board collides (mask 4/2 vs 2/4, terrain is 1/1).
            nconmax=256,
            njmax=1024,
        ),
        decimation=3,  # gym env: frame_skip 3 x timestep 0.002
        episode_length_s=6.0,  # gym env: 1000 steps x 0.006 s
    )


def labyrinth_env_cfg(play: bool = False):
    cfg = _make_labyrinth_cfg()
    if play:
        cfg.episode_length_s = 1e10
    return cfg
