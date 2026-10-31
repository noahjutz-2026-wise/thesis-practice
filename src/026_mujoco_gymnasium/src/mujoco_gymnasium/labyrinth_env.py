import json
from pathlib import Path

import gymnasium
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from gymnasium.spaces import Space
from numpy.typing import NDArray
from scipy.spatial import KDTree

WAYPOINTS_PATH = (
    Path(__file__).parents[3]
    / "023_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
)


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
    ):
        observation_space = gymnasium.spaces.Box(-1, 1, shape=(4,))
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

        self.ix = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_x")
        self.iy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_y")
        with open(WAYPOINTS_PATH, "r") as f:
            waypoints = json.load(f)["waypoints"]
            self.waypoints = np.asarray(waypoints)
            self.waypoints_tree = KDTree(self.waypoints)
        self.progress = 0

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        ctrl = np.zeros(self.model.nu)
        ctrl[self.ix] = action[0]
        ctrl[self.iy] = action[1]
        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        bp = self.data.sensor("sensor_ball_pos_wrt_board").data

        # reward
        dist, i = self.waypoints_tree.query(bp[:-1])
        delta = i - self.progress
        self.progress = i
        reward = np.float64(delta)

        # terminated
        is_offpath = dist > 0.02
        is_jumped = bp[-1] > 0.02
        terminated = is_offpath or is_jumped

        truncated = False  # TODO
        info = {}  # TODO
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def reset_model(self) -> NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        rx = self.data.sensor("sensor_board_rot_x").data
        ry = self.data.sensor("sensor_board_rot_y").data
        bp = self.data.sensor("sensor_ball_pos_wrt_board").data
        obs = np.concatenate([rx, ry, bp[:-1]])
        return obs
