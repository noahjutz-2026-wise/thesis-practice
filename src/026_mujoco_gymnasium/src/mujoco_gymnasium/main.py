from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
from mujoco import mjx

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def _dbg_render(mj_model, mj_data, renderer):
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)

    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    while mj_data.time < duration:
        mujoco.mj_step(mj_model, mj_data)
        if len(frames) < mj_data.time * framerate:
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    return frames


def main():
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)
    ix = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_x")
    iy = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_y")

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    frames = _dbg_render(mj_model, mj_data, renderer)
    for frame in frames:
        cv2.imshow("debug", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)
