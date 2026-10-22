import mujoco
from PIL import Image


def scene() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(
        pos=[0, 0, 5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        rgba=[1, 0, 0, 1],
        pos=[0, 0, -100],
    )

    body = spec.worldbody.add_body(name="room", pos=[0, 0, 1], mass=1)

    for name in ["floor", "outer", "walls"]:
        spec.add_mesh(name=name, file=f"../023_build123d/{name}.stl")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)

    body.add_joint(
        name="joint_x",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        limited=True,
        range=[-10, 10],
    )
    body.add_joint(
        name="joint_y",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        limited=True,
        range=[-10, 10],
    )

    return spec
