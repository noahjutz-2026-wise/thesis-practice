from pathlib import Path

import mujoco

MUJOCO_XML = Path.home() / "Documents/repos/thesis-practice/src/022_mujoco/export.xml"


def main():
    m = mujoco.MjModel.from_xml_path(str(MUJOCO_XML))
    d = mujoco.MjData(m)
