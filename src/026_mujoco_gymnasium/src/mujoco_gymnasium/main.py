from pathlib import Path

import cv2
import jax
import mujoco
import mujoco.viewer
from mujoco import mjx

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def _dbg_render(mj_model, mj_data, renderer, *_sink):
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


def _dbg_render_gpu(mj_model, mj_data, renderer, mjx_model, mjx_data):
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)
    jit_step = jax.jit(mjx.step)

    frames = []
    mujoco.mj_resetData(mj_model, mj_data)
    mjx_data = mjx.put_data(mj_model, mj_data)
    while mjx_data.time < duration:
        print(mjx_data.time)
        mjx_data = jit_step(mjx_model, mjx_data)
        print(mjx_data.time)
        if len(frames) < mjx_data.time * framerate:
            mj_data = mjx.get_data(mj_model, mjx_data)
            renderer.update_scene(mj_data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

    return frames


def main():
    mj_model = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    mj_data = mujoco.MjData(mj_model)

    xml = """
    <mujoco>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    # Make model, data, and renderer
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    frames = _dbg_render_gpu(mj_model, mj_data, renderer, mjx_model, mjx_data)
    for frame in frames:
        cv2.imshow("debug", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(2)
