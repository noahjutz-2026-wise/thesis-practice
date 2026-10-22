import mujoco
from PIL import Image


def scene() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 5])

    body = spec.worldbody.add_body(name="room", pos=[0, 0, 1])
    body.add_freejoint()

    for name in ["floor", "outer", "walls"]:
        spec.add_mesh(name=name, file=f"../023_build123d/{name}.stl")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)

    return spec
