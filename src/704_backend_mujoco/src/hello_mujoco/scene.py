from os.path import basename
from pathlib import Path

import mujoco
import numpy as np
from mujoco.rendering.classic.gl_context import os

MESH_PATH = Path(__file__).parents[3] / "706_maze_build123d"

X = 0.290
Y = 0.245
Z = 0.022


def scene() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(
        pos=[0, 0, 5],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.4, 0.4, 0.4],  # lower = dimmer
        specular=[0, 0, 0],  # no highlights
        ambient=[0.01, 0.01, 0.01],
    )
    spec.option.timestep = 0.002

    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        rgba=[1, 1, 1, 0.1],
        pos=[0, 0, -0.15],
    )

    m = 0.165
    maze = spec.worldbody.add_body(
        name="board",
        pos=[0, 0, 0],
        mass=m,
        explicitinertial=True,
        ipos=[0, 0, 0],
        inertia=[
            m * (Y**2 + Z**2) / 12,
            m * (X**2 + Z**2) / 12,
            m * (X**2 + Y**2) / 12,
        ],
    )

    colors = {
        "floor": [0.929, 0.796, 0.624, 1],
        "walls": [0.388, 0.141, 0, 1],
        "outer": [0.929, 0.796, 0.624, 1],
        "ball": [0.122, 0.58, 1, 1],
    }

    for part in ["floor", "outer", "walls"]:
        base_path = MESH_PATH / f"{part}_parts"
        for p in os.listdir(base_path):
            name = basename(p)
            spec.add_mesh(
                name=name, file=str(base_path / p), scale=[0.001, 0.001, 0.001]
            )
            maze.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=colors[part]
            )

    maze.add_site(
        name="imu_site",
        pos=np.array([-X / 2, -Y / 2, -0.002]) + np.array([0.005, 0.005, 0]),
        quat=[0, 1, 0, 0],
        size=[0.002, 0.002, 0.002],
    )

    maze.add_joint(
        name="joint_x",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        limited=True,
        range=[-10, 10],
        armature=0.005,
        frictionloss=0.01,
    )
    maze.add_joint(
        name="joint_y",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        limited=True,
        range=[-10, 10],
        armature=0.005,
        frictionloss=0.01,
    )

    ball = spec.worldbody.add_body(name="ball", pos=[0.01, 0.100, 0.01])
    ball.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.007], mass=0.005, rgba=colors["ball"]
    )
    ball.add_freejoint(name="joint_ball_free")

    ball.add_camera(
        name="ball_cam",
        pos=[0.0, 0.0, 0.5],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
    )

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

    spec.add_sensor(
        type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
        name="sensor_ball_vel_wrt_world",
        objtype=mujoco.mjtObj.mjOBJ_XBODY,
        objname="ball",
    )

    spec.add_sensor(
        name="sensor_imu_acc",
        type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu_site",
    )

    spec.add_sensor(
        name="sensor_imu_gyro",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu_site",
    )

    for ax in ("x", "y"):
        spec.add_actuator(
            type=mujoco.mjtActuator.mjACTUATOR_VELOCITY,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            name=f"motor_{ax}",
            target=f"joint_{ax}",
            ctrlrange=[-1, 1],
            gainprm=[0.25] + [0] * 9,
        )
    return spec
