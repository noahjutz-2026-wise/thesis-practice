from hello_mujoco.scene import scene


def export():
    s = scene()
    with open("export.xml", "w") as f:
        f.write(s.to_xml())
