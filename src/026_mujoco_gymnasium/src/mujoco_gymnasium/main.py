import math
from pathlib import Path

import mujoco
import mujoco.viewer

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def main():
    m = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    d = mujoco.MjData(m)
    ix = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_x")
    iy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_y")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while True:
            d.ctrl[ix] = math.sin(d.time) * 0.005
            d.ctrl[iy] = math.cos(d.time) * 0.005
            mujoco.mj_step(m, d)
            viewer.sync()
        while True:
            pass
