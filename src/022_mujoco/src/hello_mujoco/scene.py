import mujoco


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

    maze = spec.worldbody.add_body(name="board", pos=[0, 0, 1], mass=1)

    for name in ["floor", "outer", "walls"]:
        spec.add_mesh(name=name, file=f"../023_build123d/{name}.stl")
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

    ball = spec.worldbody.add_body(name="ball", pos=[0, 0, 10], mass=0.07)
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[7])
    ball.add_freejoint()

    return spec
