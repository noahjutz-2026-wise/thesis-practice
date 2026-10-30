import math

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from gymnasium.spaces import Space
from numpy.typing import NDArray


class LabyrinthEnv(MujocoEnv):
    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        observation_space: Space | None = None,
        render_mode: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        camera_id: int | None = None,
        camera_name: str | None = None,
        default_camera_config: dict[str, float] | None = None,
        max_geom: int = 1000,
        visual_options: dict[int, bool] | None = None,
    ):
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
            visual_options,
        )

        self.ix = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_x")
        self.iy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_y")

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64], np.float64, bool, bool, dict[str, np.float64]]:
        ctrl = np.zeros(self.model.nu)
        ctrl[self.ix] = action[0]
        ctrl[self.iy] = action[1]
        self.do_simulation(ctrl, self.frame_skip)

        rx = self.data.sensor("sensor_board_rot_x").data
        ry = self.data.sensor("sensor_board_rot_y").data
        bp = self.data.sensor("sensor_ball_pos").data

        obs = np.concatenate([rx, ry, bp[:-1]])
        reward = np.float64(0)  # TODO
        terminated = bp[-1] < -0.05
        truncated = False  # TODO
        info = {}  # TODO
        return obs, reward, terminated, truncated, info

    def reset_model(self) -> NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return np.zeros(())  # TODO
