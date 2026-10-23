import mujoco
import numpy as np
from mujoco import sysid

MODEL_XML = "../022_mujoco/export.xml"

spec = mujoco.MjSpec.from_file(MODEL_XML)
model = spec.compile()
data = mujoco.MjData(model)

params = sysid.ParameterDict()

# Dynamically add stiffness, damping, armature, and friction params for both X and Y axes
for axis in ("x", "y"):
    # 1. Motor Actuator Stiffness (Gain / kp)
    params.add(
        sysid.Parameter(
            f"stiffness_{axis}",
            nominal=100.0,
            min_value=10.0,
            max_value=500.0,
            modifier=lambda model, p, a=axis: setattr(
                model.actuator(f"motor_{a}"),
                "gainprm",
                [p.value[0]] + list(model.actuator(f"motor_{a}").gainprm[1:]),
            ),
        )
    )

    # 2. Motor Actuator Damping (Velocity Gain / kv)
    params.add(
        sysid.Parameter(
            f"damping_{axis}",
            nominal=10.0,
            min_value=0.1,
            max_value=50.0,
            modifier=lambda model, p, a=axis: setattr(
                model.actuator(f"motor_{a}"),
                "biasprm",
                [0.0, -p.value[0], 0.0]
                + list(model.actuator(f"motor_{a}").biasprm[3:]),
            ),
        )
    )

    # 1. Joint Armature (Scalar float)
    params.add(
        sysid.Parameter(
            f"armature_{axis}",
            nominal=0.01,
            min_value=0.001,
            max_value=0.1,
            modifier=lambda model, p, a=axis: setattr(
                model.joint(f"joint_{a}"), "armature", float(p.value[0])
            ),
        )
    )

    # 2. Joint Friction Loss (Scalar float)
    params.add(
        sysid.Parameter(
            f"friction_{axis}",
            nominal=0.1,
            min_value=0.0,
            max_value=2.0,
            modifier=lambda model, p, a=axis: setattr(
                model.joint(f"joint_{a}"), "frictionloss", float(p.value[0])
            ),
        )
    )

initial_state = sysid.create_initial_state(model, data.qpos, data.qvel, data.act)

control_ts = sysid.TimeSeries.from_names(
    np.arange(1000) * 0.001, np.ones((1000, 2)), model
)
sensor_ts = sysid.TimeSeries.from_control_names(
    np.arange(1000) * 0.001, np.ones((1000, 2)), model
)

ms = sysid.ModelSequences("ms", spec, "measured", initial_state, control_ts, sensor_ts)

residual_fn = sysid.build_residual_fn(models_sequences=[ms])
opt_params, opt_result = sysid.optimize(initial_params=params, residual_fn=residual_fn)
sysid.save_results("results/", [ms], params, opt_params, opt_result, residual_fn)


def main():
    pass
