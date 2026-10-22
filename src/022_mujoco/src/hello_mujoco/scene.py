from os.path import basename
from pathlib import Path

import mujoco
from mujoco.rendering.classic.gl_context import os

MESH_PATH = Path("../023_build123d/")


def scene() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(
        pos=[0, 0, 5], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        rgba=[1, 0, 0, 1],
        pos=[0, 0, -0.05],
    )

    maze = spec.worldbody.add_body(name="board", pos=[0, 0, 0])

    for name in ["floor", "outer", "walls"]:
        base_path = MESH_PATH / f"{name}_parts"
        for p in os.listdir(base_path):
            name = basename(p)
            spec.add_mesh(
                name=name, file=str(base_path / p), scale=[0.001, 0.001, 0.001]
            )
            maze.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)

    maze.add_joint(
        name="joint_x",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        limited=True,
        range=[-10, 10],
    )
    maze.add_joint(
        name="joint_y",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        limited=True,
        range=[-10, 10],
    )

    ball = spec.worldbody.add_body(name="ball", pos=[0.1, -0.08, 0.07])
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.007], mass=0.005)
    ball.add_freejoint()

    return spec
