"""Labyrinth (ball-on-tilting-board) environment for mjlab.

Mirrors the gymnasium env in src/702_gym_mujoco but keeps only the core
behavior: tilt the board with two motors to roll the ball along the waypoint
path. Reference (not copy) of the XML from src/704_backend_mujoco.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.envs.mdp.events import resolve_env_ids
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
FALL_Z = -0.004  # ball world z below this counts as fallen; resting z is
# +0.0066, so this triggers the moment the ball tunnels through the floor
# (deep-contact failure at wall bases) instead of letting it wedge inside
# the slab, which used to freeze episodes in an unrecoverable state

# --- waypoint path (loaded once at import) ----------------------------------
#
# The layout's reference path is INFEASIBLE: 11% of it runs inside the ball
# capture zone of a hole (capture = hole r 7.75mm + ball r 7mm). The path is
# repaired at import: repel points away from holes (>= RC), from walls
# (>= wall half-width + ball r + margin) and inside the outer frame, then
# smooth. The repaired path defines progress, goal and lookahead targets.

_HOLE_R = 0.00775     # HOLE_DIAMETER 15.5mm / 2 (maze.py)
_BALL_R = 0.007
_RC = _HOLE_R + _BALL_R + 0.003        # min clearance from hole centers
_WALL_HALF = 0.0025                    # WALL_WIDTH 5mm / 2
_RW = _WALL_HALF + _BALL_R + 0.002     # min clearance from wall centerlines
_XLIM, _YLIM = 0.145 - 0.0075, 0.1225 - 0.0075  # inner edge of outer frame
_RBO = _BALL_R + 0.002                 # and the ball margin inside it


def _seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to segment ab."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def _repair_path(wps: np.ndarray, holes: np.ndarray, walls: list) -> np.ndarray:
    segs = np.diff(wps, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    pts = []
    for i, L in enumerate(lens):
        n = max(1, int(L / 0.002))
        pts.extend(wps[i] + t * segs[i] for t in np.linspace(0, 1, n, endpoint=False))
    pts.append(wps[-1])
    pts = np.asarray(pts)

    for _ in range(300):
        moved = 0.0
        # holes: radial push to RC
        dists = np.linalg.norm(pts[:, None, :] - holes[None, :, :], axis=2)
        near = dists.min(axis=1)
        hi = dists.argmin(axis=1)
        bad = near < _RC
        if bad.any():
            push = pts[bad] - holes[hi[bad]]
            push /= np.linalg.norm(push, axis=1, keepdims=True).clip(1e-9)
            target = holes[hi[bad]] + push * _RC
            moved += np.abs(target - pts[bad]).sum()
            pts[bad] = target
        # walls: push out of the wall+ball margin
        for a, b in walls:
            d = _seg_dist(pts, a, b)
            badw = d < _RW
            if badw.any():
                ab = b - a
                t = np.clip(((pts[badw] - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                proj = a + t[:, None] * ab
                push = pts[badw] - proj
                nrm = np.linalg.norm(push, axis=1, keepdims=True).clip(1e-9)
                target = proj + push / nrm * _RW
                moved += np.abs(target - pts[badw]).sum()
                pts[badw] = target
        # outer frame margin
        pts[:, 0] = pts[:, 0].clip(-_XLIM + _RBO, _XLIM - _RBO)
        pts[:, 1] = pts[:, 1].clip(-_YLIM + _RBO, _YLIM - _RBO)
        # smoothing (keep endpoints fixed)
        smooth = 0.25 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0)) - pts
        smooth[0] = smooth[-1] = 0.0
        pts += 0.3 * smooth
        if moved < 1e-7:
            break
    # final check
    dists = np.linalg.norm(pts[:, None, :] - holes[None, :, :], axis=2).min(axis=1)
    if (dists < _HOLE_R + _BALL_R).any():
        raise RuntimeError(f"path repair failed: {(dists < _HOLE_R + _BALL_R).sum()} points still capturable")
    return pts


def _resample(pts: np.ndarray, spacing: float = 0.005) -> np.ndarray:
    segs = np.diff(pts, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    total = lens.sum()
    n = max(2, int(total / spacing))
    cum = np.concatenate([[0.0], np.cumsum(lens)]) / total
    out = []
    for s in np.linspace(0.0, 1.0, n):
        i = np.clip(np.searchsorted(cum, s) - 1, 0, len(lens) - 1)
        t = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
        out.append(pts[i] + t * segs[i])
    return np.asarray(out)


def _route_path(spawn: np.ndarray, goal: np.ndarray, holes: np.ndarray,
                walls: list, step: float = 0.001) -> np.ndarray:
    """Shortest feasible route on a 1mm grid: cells must clear holes by RC,
    walls by RW, and stay inside the outer frame margin. 8-connected Dijkstra.
    Wall clearance allows ball<->wall CONTACT (9.6mm = half-width + ball r
    + 0.1mm slack); hole clearance keeps a momentum-crossing margin."""
    step = 0.001
    # Effective ball radius at wall-top height: walls are 6 mm tall, ball
    # center ~7 mm, so at a pinch the sphere's neck (~6.9 mm) is what must
    # fit — 2.5 (wall half) + 6.9 = 9.4 mm. Grid at 0.5 mm to keep the
    # ~0.2 mm free slivers of exactly-ball-sized corridors.
    step = 0.0005
    wall_clear = _WALL_HALF + 0.0069
    xs = np.arange(-_XLIM + _RBO, _XLIM - _RBO + step, step)
    ys = np.arange(-_YLIM + _RBO, _YLIM - _RBO + step, step)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (C, 2)

    free = np.ones(len(grid), dtype=bool)
    hole_soft = np.zeros(len(grid))
    if len(holes):
        hd = np.linalg.norm(grid[:, None, :] - holes[None, :, :], axis=2)
        hd = hd.min(axis=1)
        # Empirical ball capture: the ball bridges a hole rim down to ~8.5mm
        # from the hole center (it survived 10mm in training); below that it
        # drops. Hard-ban the fatal zone; 8.5-14mm stays a soft cost (tight
        # wall-adjacent lanes the ball can thread hugging the wall).
        free &= (hd > 0.0085)
        hole_soft = np.clip((0.014 - hd) / 0.006, 0.0, 1.0)
    for a, b in walls:
        free &= (_seg_dist(grid, a, b) > wall_clear)

    def nearest_free(p):
        d = np.linalg.norm(grid - p, axis=1)
        d[~free] = np.inf
        return int(d.argmin())

    start, end = nearest_free(spawn), nearest_free(goal)
    import heapq
    nx, ny = len(xs), len(ys)
    idx = np.arange(len(grid)).reshape(ny, nx)
    dist = np.full(len(grid), np.inf)
    prev = np.full(len(grid), -1, dtype=np.int64)
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u]:
            continue
        if u == end:
            break
        row, col = divmod(u, nx)  # u = row*nx + col (row-major, rows = ys)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                c2, r2 = col + dx, row + dy
                if not (0 <= c2 < nx and 0 <= r2 < ny):
                    continue
                v = int(idx[r2, c2])
                if not free[v]:
                    continue
                w = du + (1.414 if dx and dy else 1.0) * (1.0 + 12.0 * hole_soft[v])
                if w < dist[v]:
                    dist[v] = w
                    prev[v] = u
                    heapq.heappush(pq, (w, v))
    if dist[end] == np.inf:
        raise RuntimeError("no feasible route from spawn to goal on the free-space grid")
    path = [end]
    while path[-1] != start:
        path.append(int(prev[path[-1]]))
    return grid[np.array(path)[::-1]]


with open(LAYOUT_PATH, "r") as f:
    _layout = json.load(f)
_wp0 = np.asarray(_layout["waypoints"], dtype=np.float64)
_holes = np.asarray(_layout["holes"], dtype=np.float64)
_wall_segs = []
for a, b, y in _layout["walls_v"]:  # vertical wall at x=y, y-range a..b
    _wall_segs.append((np.array([y, a]), np.array([y, b])))
for a, b, y in _layout["walls_h"]:  # horizontal wall at y=y, x-range a..b
    _wall_segs.append((np.array([a, y]), np.array([b, y])))

# The layout path is illegal in places: some segments run 1.5 mm INSIDE
# walls (ball center needs 9.5 mm from wall centerlines). Quasi-static hole
# avoidance is impossible (no hole-safe route exists), so the path is
# rerouted for WALL legality only; holes are crossed with momentum, which a
# policy can learn (proven: ladder66long crosses the wp67-hole stretch).
_WAYPOINTS = torch.tensor(
    _resample(_route_path(_wp0[0], _wp0[-1], _holes, _wall_segs)),
    dtype=torch.float32,
)

_SEG = _WAYPOINTS[1:] - _WAYPOINTS[:-1]
_SEG_LEN = _SEG.norm(dim=-1)
_CUM = torch.cat([torch.zeros(1), torch.cumsum(_SEG_LEN, dim=0)]) / _SEG_LEN.sum()

# Dense resampling of the path for lookahead targets.
_DENSE_N = 1000
_u = torch.linspace(0.0, 1.0, _DENSE_N)
_seg_idx = torch.clamp(torch.searchsorted(_CUM, _u, right=True) - 1, 0, len(_SEG) - 1)
_t = (_u - _CUM[_seg_idx]) / (_CUM[_seg_idx + 1] - _CUM[_seg_idx]).clamp_min(1e-9)
_DENSE = _WAYPOINTS[_seg_idx] + _t.unsqueeze(-1) * _SEG[_seg_idx]

with open(LAYOUT_PATH, "r") as f:
    _HOLES = torch.tensor(json.load(f)["holes"], dtype=torch.float32)

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


def goal_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Large positive reward on the step the ball reaches the goal."""
    return env.termination_manager.get_term("goal").float()


def fall_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Negative reward on the step the ball falls off the board."""
    return env.termination_manager.get_term("fallen").float()


def respawn_on_fall(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Checkpoint respawn instead of episode-ending fall.

    When the ball drops below FALL_Z it is teleported back to the last
    passed waypoint (zero velocity, progress anchored there) and the
    episode continues. The returned penalty (weighted by the fall-penalty
    weight) is the only cost besides lost time.
    """
    _ensure_progress(env)
    dev = env.device
    ball: Entity = env.scene["ball"]
    z = ball.data.body_link_pos_w[:, 0, 2]
    fallen = z < FALL_Z
    if not fallen.any():
        return torch.zeros(env.num_envs, device=dev)
    cum = _CUM.to(dev)
    p = env._labyrinth_progress[fallen]
    k = (torch.searchsorted(cum, p, right=True) - 1).clamp(0, len(cum) - 2)
    wps = _WAYPOINTS.to(dev)
    rp = wps[k]  # (f, 2) respawn positions
    idx = torch.nonzero(fallen).flatten()
    n = len(idx)
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = rp[:, 0]
    pose[:, 1] = rp[:, 1]
    pose[:, 2] = BALL_START_Z
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids=idx)
    ball.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=dev), env_ids=idx
    )
    # progress is a RATCHET (kept at its max): falls cost only time and the
    # respawn penalty, so attempting a risky crossing is never net-negative.
    return torch.ones(env.num_envs, device=dev)


def labyrinth_progress_reward(
    env: ManagerBasedRlEnv, time_penalty: float = 1e-4
) -> torch.Tensor:
    """Reward for claiming NEW path arc (ratchet); re-traversal after a
    respawn earns nothing. Falls therefore cost only time."""
    _ensure_progress(env)
    newp, _ = _project_onto_path(env, _ball_xy(env))
    reward = (newp - env._labyrinth_progress).clamp_min(0.0) - time_penalty
    env._labyrinth_progress = torch.maximum(env._labyrinth_progress, newp)
    return reward


def ball_fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    z = ball.data.body_link_pos_w[:, 0, 2]
    return z < FALL_Z


def goal_reached(env: ManagerBasedRlEnv) -> torch.Tensor:
    _ensure_progress(env)
    return env._labyrinth_progress >= 1.0 - 1e-6


def ball_target_vecs(
    env: ManagerBasedRlEnv, normalize: bool = False, ctrl_frame: bool = False
) -> torch.Tensor:
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
    vecs = targets - bp.unsqueeze(1)
    if normalize:
        vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if ctrl_frame:
        vecs = _rot90(vecs)
    return vecs.reshape(bp.shape[0], -1)


def approach_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Dense shaping: negative distance from ball to the 0.02-lookahead target."""
    _ensure_progress(env)
    dev = env.device
    bp = _ball_xy(env)
    dense = _DENSE.to(dev)
    idx = ((env._labyrinth_progress + 0.02).clamp(0.0, 1.0) * (_DENSE_N - 1)).long()
    target = dense[idx]
    return -(target - bp).norm(dim=-1)


def _rot90(v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors into the board-tilt (action) frame.

    Empirical mapping: ctrl_x=+ rolls the ball toward -y, ctrl_y=+ toward
    +x, so ctrl = (-w_y, w_x) for desired world motion w.
    """
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)


def ball_vel_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ball linear velocity, rotated into the tilt (action) frame."""
    ball: Entity = env.scene["ball"]
    vel = ball.data.body_link_lin_vel_w[:, 0, :2]
    return _rot90(vel)


def holes_obs(env: ManagerBasedRlEnv, k: int = 3) -> torch.Tensor:
    """Relative position (dx, dy) and distance of the k nearest hole centers,
    rotated into the tilt (action) frame."""
    bp = _ball_xy(env)  # (n, 2)
    holes = _HOLES.to(bp.device)  # (H, 2)
    d = (bp.unsqueeze(1) - holes.unsqueeze(0)).norm(dim=-1)  # (n, H)
    idx = d.topk(k, dim=1, largest=False).indices  # k nearest
    nearest = holes[idx]  # (n, k, 2)
    rel = _rot90(nearest - bp.unsqueeze(1))
    dist = d.gather(1, idx).unsqueeze(-1)
    return torch.cat([rel, dist], dim=-1).reshape(bp.shape[0], -1)


def ball_z_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    return ball.data.body_link_pos_w[:, 0, 2].unsqueeze(-1)


def ball_state_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    ball: Entity = env.scene["ball"]
    pos = ball.data.body_link_pos_w[:, 0, :2]
    vel = ball.data.body_link_lin_vel_w[:, 0, :2]
    return torch.cat([pos, vel], dim=-1)


def reset_ball_at_start(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    start_wp_idx: int | None = None,
) -> None:
    """Reset the ball to the XML default start pose (or a fixed waypoint when
    start_wp_idx is given) and progress to the matching arc position."""
    _ensure_progress(env)
    env_ids = resolve_env_ids(env, env_ids)
    ball: Entity = env.scene["ball"]
    n = len(env_ids)
    dev = env.device

    if start_wp_idx is None:
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0] = BALL_START_X
        pose[:, 1] = BALL_START_Y
        pose[:, 2] = BALL_START_Z
        pose[:, 3] = 1.0
        progress = 0.0
    else:
        wp = _WAYPOINTS.to(dev)[start_wp_idx]
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0] = wp[0]
        pose[:, 1] = wp[1]
        pose[:, 2] = BALL_START_Z
        pose[:, 3] = 1.0
        progress = float(_CUM[start_wp_idx])
    ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
    env._labyrinth_progress[env_ids] = progress


# --- env cfg -----------------------------------------------------------------


def _make_labyrinth_cfg(
    fall_penalty_weight: float = 0.0,
    action_scale: float = 1.0,
    ball_z_obs: bool = False,
    time_penalty_weight: float = 1e-4,
    normalize_tvec: bool = False,
    approach_weight: float = 0.0,
    holes_obs_k: int = 0,
    goal_bonus_weight: float = 0.0,
    start_wp_idx: int | None = None,
    ctrl_frame_obs: bool = False,
    progress_weight: float = 1.0,
    episode_steps: int = 1000,
    respawn: bool = False,
) -> ManagerBasedRlEnvCfg:
    tilt_terms = {
        "tilt_pos": ObservationTermCfg(
            func=joint_pos_rel, params={"asset_cfg": _BOARD_CFG}
        ),
        "tilt_vel": ObservationTermCfg(
            func=joint_vel_rel, params={"asset_cfg": _BOARD_CFG}
        ),
        "ball_state": ObservationTermCfg(func=ball_state_obs),
        "target_vecs": ObservationTermCfg(
            func=ball_target_vecs, params={"normalize": normalize_tvec}
        ),
    }
    if ctrl_frame_obs:
        tilt_terms["ball_state"] = ObservationTermCfg(func=ball_vel_obs)
        tilt_terms["target_vecs"] = ObservationTermCfg(
            func=ball_target_vecs,
            params={"normalize": normalize_tvec, "ctrl_frame": True},
        )
    if ball_z_obs:
        tilt_terms["ball_z"] = ObservationTermCfg(func=ball_z_obs)
    if holes_obs_k:
        tilt_terms["holes"] = ObservationTermCfg(
            func=holes_obs, params={"k": holes_obs_k}
        )
    observations = {
        "actor": ObservationGroupCfg(tilt_terms),
        "critic": ObservationGroupCfg({**tilt_terms}),
    }
    actions = {
        "effort": JointEffortActionCfg(
            entity_name="board",
            actuator_names=("joint_x", "joint_y"),
            scale=action_scale,
        ),
    }
    rewards = {
        "progress": RewardTermCfg(
            func=labyrinth_progress_reward,
            weight=progress_weight,
            params={"time_penalty": time_penalty_weight},
        ),
    }
    if fall_penalty_weight != 0.0:
        if respawn:
            rewards["fall_penalty"] = RewardTermCfg(
                func=respawn_on_fall, weight=fall_penalty_weight
            )
        else:
            rewards["fall_penalty"] = RewardTermCfg(
                func=fall_penalty, weight=fall_penalty_weight
            )
    if approach_weight != 0.0:
        rewards["approach"] = RewardTermCfg(
            func=approach_reward, weight=approach_weight
        )
    if goal_bonus_weight != 0.0:
        rewards["goal_bonus"] = RewardTermCfg(func=goal_bonus, weight=goal_bonus_weight)
    terminations = {
        "time_out": TerminationTermCfg(func=time_out, time_out=True),
        "goal": TerminationTermCfg(func=goal_reached),
    }
    if not respawn:
        terminations["fallen"] = TerminationTermCfg(func=ball_fallen)
    events = {
        "reset_ball": EventTermCfg(
            func=reset_ball_at_start,
            mode="reset",
            params={"start_wp_idx": start_wp_idx},
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
        episode_length_s=episode_steps * 0.006,
    )


def labyrinth_env_cfg(
    play: bool = False,
    fall_penalty_weight: float = 0.0,
    action_scale: float = 1.0,
    ball_z_obs: bool = False,
    time_penalty_weight: float = 1e-4,
    normalize_tvec: bool = False,
    approach_weight: float = 0.0,
    holes_obs_k: int = 0,
    goal_bonus_weight: float = 0.0,
    start_wp_idx: int | None = None,
    ctrl_frame_obs: bool = False,
    progress_weight: float = 1.0,
    episode_steps: int = 1000,
    respawn: bool = False,
):
    cfg = _make_labyrinth_cfg(
        fall_penalty_weight=fall_penalty_weight,
        action_scale=action_scale,
        ball_z_obs=ball_z_obs,
        time_penalty_weight=time_penalty_weight,
        normalize_tvec=normalize_tvec,
        approach_weight=approach_weight,
        holes_obs_k=holes_obs_k,
        goal_bonus_weight=goal_bonus_weight,
        start_wp_idx=start_wp_idx,
        ctrl_frame_obs=ctrl_frame_obs,
        progress_weight=progress_weight,
        episode_steps=episode_steps,
        respawn=respawn,
    )
    if play:
        cfg.episode_length_s = 1e10
    return cfg


def labyrinth_ppo_runner_cfg(
    experiment_name: str = "labyrinth",
    entropy_coef: float = 0.01,
    init_std: float = 1.0,
    max_iterations: int = 500,
    gamma: float = 0.99,
    lr: float = 1.0e-3,
    desired_kl: float = 0.01,
):
    from mjlab.rl.config import (
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=entropy_coef,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=lr,
            schedule="adaptive",
            gamma=gamma,
            lam=0.95,
            desired_kl=desired_kl,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=max_iterations,
    )
