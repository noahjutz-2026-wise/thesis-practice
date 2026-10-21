import mujoco
from PIL import Image


def main():
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 5])
    for name in ["floor", "outer", "walls"]:
        spec.add_mesh(name=name, file=f"../023_build123d/{name}.stl")
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    Image.fromarray(renderer.render()).save("scene.png")


if __name__ == "__main__":
    main()
