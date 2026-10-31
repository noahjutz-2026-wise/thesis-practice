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
    spec.option.timestep = 0.002

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        rgba=[1, 0, 0, 1],
        pos=[0, 0, -0.15],
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

    ball = spec.worldbody.add_body(name="ball", pos=[0.01, 0.100, 0.01])
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.007], mass=0.005)
    ball.add_freejoint(name="joint_ball_free")

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
        name="sensor_board_rot_x",
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        objname="joint_x",
    )

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
        name="sensor_board_rot_y",
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        objname="joint_y",
    )

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
        name="sensor_board_rot_quat",
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname="board",
    )

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        name="sensor_ball_pos_wrt_world",
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="ball",
    )

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        name="sensor_ball_pos_wrt_board",
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="ball",
        reftype=mujoco.mjtObj.mjOBJ_XBODY,
        refname="board",
    )

    spec.add_actuator(
        type=mujoco.mjtActuator.mjACTUATOR_VELOCITY,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name="motor_x",
        target="joint_x",
        ctrlrange=[-1, 1],
    )

    spec.add_actuator(
        type=mujoco.mjtActuator.mjACTUATOR_VELOCITY,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name="motor_y",
        target="joint_y",
        ctrlrange=[-1, 1],
    )
    return spec
