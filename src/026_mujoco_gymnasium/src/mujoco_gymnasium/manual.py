import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

import mujoco_gymnasium.labyrinth_env as labyrinth_env

MUJOCO_XML = Path(__file__).parents[3] / "022_mujoco/export.xml"
SPEED = 0.15
DT = 0.006
KEYS = {
    glfw.KEY_LEFT: (1, -1),
    glfw.KEY_RIGHT: (1, 1),
    glfw.KEY_DOWN: (0, -1),
    glfw.KEY_UP: (0, 1),
}


def main():
    env = labyrinth_env.LabyrinthEnv(str(MUJOCO_XML), 3)
    env.reset()
    target = np.zeros(2)
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    win = glfw.create_window(480, 480, "manual", None, None)
    glfw.make_context_current(win)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = 0.35
    cam.elevation = -90
    cam.azimuth = 90
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(env.model, 1000)
    con = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, 480, 480)
    while not glfw.window_should_close(win):
        glfw.poll_events()
        for key, (axis, sign) in KEYS.items():
            if glfw.get_key(win, key) == glfw.PRESS:
                target[axis] += sign * SPEED * DT
        target = np.clip(target, -0.15, 0.15)
        env.data.qpos[:2] = target
        env.data.qvel[:2] = 0.0
        env.do_simulation(np.zeros(env.model.nu), env.frame_skip)
        bp = env.data.sensor("sensor_ball_pos_wrt_board").data
        dist, i = env.waypoints_tree.query(bp[:-1])
        env.progress = i
        if bp[2] < -0.05 or dist > 0.05:
            env.reset()
            target = np.zeros(2)
        mujoco.mjv_updateScene(
            env.model, env.data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, con)
        glfw.swap_buffers(win)
        time.sleep(DT)
    glfw.terminate()


if __name__ == "__main__":
    main()
