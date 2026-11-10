import json
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
import shapely
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from numpy.typing import NDArray
from shapely.ops import substring

WAYPOINTS_PATH = (
    Path(__file__).parents[3]
    / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
)

WINDOW = 0.03  # normalized progress +/- used for reward subpath and highlight


class LabyrinthEnv(MujocoEnv):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        render_mode: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict[str, float] | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] | None = None,
        random_starts: bool = True,
    ):
        observation_space = gymnasium.spaces.Box(-1, 1, shape=(8,))
        super().__init__(
            model_path,
            frame_skip,
            observation_space,
            render_mode,
            width,
            height,
            camera_id,
            camera_name,
            default_camera_config,
            max_geom,
            {} if visual_options is None else visual_options,
        )

        self.random_starts = random_starts
        self.ix = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_x")
        self.iy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_y")
        with open(WAYPOINTS_PATH, "r") as f:
            waypoints = json.load(f)["waypoints"]
            self.waypoints = np.asarray(waypoints)
            self.line = shapely.geometry.LineString(self.waypoints)

        self.progress = 0
        self.t = 0
        self.stall_idx = 0
        self.bp = np.zeros(())
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ball_cam")
        self.track_renderer = mujoco.Renderer(self.model, height=64, width=64)
        self.buf = np.zeros((64, 64, 3), np.uint8)

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        ctrl = np.zeros(self.model.nu)
        ctrl[self.ix] = action[0]
        ctrl[self.iy] = action[1]
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        bp = self.data.sensor("sensor_ball_pos_wrt_board").data.copy()

        # reward
        penalty = 0.0001
        xy = shapely.geometry.Point(bp[:-1])
        lo, hi = max(0, self.progress - WINDOW), min(1, self.progress + WINDOW)
        sub = substring(self.line, lo, hi, normalized=True)
        local = sub.project(xy, normalized=True)
        newp = lo + (hi - lo) * local
        dist = sub.distance(xy)
        delta = newp - self.progress
        self.progress = newp
        reward = delta - penalty
        reward = np.float64(reward)

        # terminated
        if np.linalg.norm(bp - self.bp) < 1e-6:
            self.stall_idx += 1
        else:
            self.stall_idx = 0
        self.bp = bp
        is_offpath = dist > 0.03
        is_fallen = bp[-1] < -0.05
        if is_fallen:
            reward = -0.1
        is_stall = self.stall_idx >= 50
        is_goal = self.progress >= 1.0
        if is_goal:
            reward = 1
        cond = np.array([is_offpath, is_fallen, is_stall, is_goal])
        terminated = bool(np.any(cond))

        truncated = self.t >= 50_000
        info = {}  # TODO
        if self.render_mode == "human":
            self._render_human()
        self.t += 1
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.t = 0
        self.progress = 0
        self.stall_idx = 0
        return super().reset(seed=seed, options=options)

    def reset_model(self) -> NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel

        if self.random_starts:
            idx = int(self.np_random.integers(len(self.waypoints)))
            self.progress = self.line.project(
                shapely.geometry.Point(self.waypoints[idx]), normalized=True
            )
            target_pos = self.waypoints[idx]
            target_pos += self.np_random.uniform(-0.002, 0.002, size=2)
            adr = self.model.joint("joint_ball_free").qposadr[0]
            qpos[adr : adr + 2] = target_pos

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        rx = self.data.sensor("sensor_board_rot_x").data
        ry = self.data.sensor("sensor_board_rot_y").data
        bp = self.data.sensor("sensor_ball_pos_wrt_board").data
        bv = self.data.sensor("sensor_ball_vel_wrt_world").data
        img = self._get_subimg().flatten()

        obs = np.concatenate([rx, ry, bp[:-1], bv[:-1], self._dir_to_target(bp), img])
        return obs.astype(np.float32)

    def _get_subimg(self) -> NDArray[np.uint8]:
        self.track_renderer.update_scene(self.data, camera="ball_cam")
        self.track_renderer.render(out=self.buf)
        # grayscale in place: (64, 64, 1) uint8, channel-order-free
        self.buf[:, :] = self.buf.mean(axis=-1, keepdims=True).astype(np.uint8)
        return self.buf[:, :, 0]

    def _dir_to_target(self, bp: NDArray[np.float64]) -> NDArray[np.float64]:
        look = 0.02
        target = self.line.interpolate(min(1.0, self.progress + look), normalized=True)
        to_target = np.array([target.x, target.y]) - bp[:-1]
        dist_t = np.linalg.norm(to_target)
        return to_target / (dist_t + 1e-8)

    def _render_human(self):
        bp = self.data.sensor("sensor_ball_pos_wrt_board").data
        dir_t = self._dir_to_target(bp)

        viewer = self.mujoco_renderer._get_viewer(self.render_mode)
        viewer.make_context_current()
        quat, mat = np.zeros(4), np.zeros(9)
        mujoco.mju_quatZ2Vec(quat, np.array([dir_t[0], dir_t[1], 0.0]))
        mujoco.mju_quat2Mat(mat, quat)

        point = self.line.interpolate(self.progress, normalized=True)

        board = self.data.body("board")
        R, p = board.xmat.reshape(3, 3), board.xpos

        def to_world(x, y):
            return p + R @ np.array([x, y, 0.008])

        pts = np.array([to_world(x, y) for x, y in self.waypoints])
        lo, hi = max(0.0, self.progress - WINDOW), min(1.0, self.progress + WINDOW)
        seglen = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        cum /= cum[-1]
        markers = []
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            z = (b - a) / np.linalg.norm(b - a)
            x_ = np.array([z[1], -z[0], 0.0])
            x_ /= np.linalg.norm(x_)
            y_ = np.cross(z, x_)
            seg_mat = np.column_stack([x_, y_, z]).ravel()

            def cyl(p0, p1, rgba):
                d = p1 - p0
                markers.append(
                    {
                        "type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                        "size": np.array([0.0015, np.linalg.norm(d) / 2, 0.0]),
                        "pos": (p0 + p1) / 2,
                        "mat": seg_mat,
                        "rgba": rgba,
                    }
                )

            blue = np.array([0.0, 0.0, 1.0, 0.4], dtype=np.float32)
            yellow = np.array([1.0, 0.9, 0.0, 1.0], dtype=np.float32)
            t0, t1 = cum[i], cum[i + 1]
            if t1 <= lo or t0 >= hi:  # fully outside window
                cyl(a, b, blue)
                continue
            f = lambda t: a + (b - a) * ((t - t0) / (t1 - t0))
            if lo > t0:
                cyl(a, f(lo), blue)
            cyl(f(max(lo, t0)), f(min(hi, t1)), yellow)
            if hi < t1:
                cyl(f(hi), b, blue)
        markers += [
            {
                "type": mujoco.mjtGeom.mjGEOM_ARROW,
                "size": np.array([0.002, 0.002, 0.03]),
                "pos": self.data.body("ball").xpos + [0, 0, 0.02],
                "mat": mat,  # flat, shape (9,)
                "rgba": np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            },
            {
                "type": mujoco.mjtGeom.mjGEOM_SPHERE,
                "size": np.array([0.005, 0.0, 0.0]),
                "pos": to_world(point.x, point.y),
                "rgba": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            },
        ]
        viewer._markers = markers
        self.render()
