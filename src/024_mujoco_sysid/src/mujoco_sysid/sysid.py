from glob import glob
from pathlib import Path

import mujoco
import numpy as np
from mujoco import sysid

MODEL_XML = "../022_mujoco/export.xml"


spec = mujoco.MjSpec.from_file(MODEL_XML)
measured_sensor_names = {"sensor_imu_acc", "sensor_imu_gyro"}

for sensor in list(spec.sensors):
    if sensor.name not in measured_sensor_names:
        spec.delete(sensor)

spec.site("imu_site").quat = (0, 0.7071, 0.7071, 0)


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
            min_value=0.1,
            max_value=5000.0,
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

DATA_GLOB = "../027_backend_real/sysid_data/*.npz"
FREQUENCY = 200
CONTROL_SCALE = 1000
ACC_SCALE = 10.0 / 8.745
GYRO_BIAS = np.array([-0.0507, 0.1054, -0.0125])

models_sequences = []


def sensor_mapping(model):
    mapping = {}
    for name in ("sensor_imu_acc", "sensor_imu_gyro"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = model.sensor_adr[sid], model.sensor_dim[sid]
        for j in range(dim):
            mapping[f"{name}[{j}]"] = (
                sysid.SignalType.MjSensor,
                np.array([adr + j]),
            )
    return mapping


for path in sorted(glob(DATA_GLOB)):
    d = np.load(path)
    t = np.arange(len(d["sensor"])) / FREQUENCY
    meas = d["sensor"].copy()
    meas[:, :3] *= ACC_SCALE
    meas[:, 3:6] -= GYRO_BIAS
    control_ts = sysid.TimeSeries.from_control_names(
        t, d["action"] / CONTROL_SCALE, model
    )
    sensor_ts = sysid.TimeSeries(
        times=t, data=meas, signal_mapping=sensor_mapping(model)
    )
    models_sequences.append(
        sysid.ModelSequences(
            Path(path).stem,
            spec,
            "measured",
            initial_state,
            control_ts,
            sensor_ts,
            allow_missing_sensors=True,
        )
    )

residual_fn = sysid.build_residual_fn(models_sequences=models_sequences)
opt_params, opt_result = sysid.optimize(initial_params=params, residual_fn=residual_fn)
sysid.save_results(
    "results/", models_sequences, params, opt_params, opt_result, residual_fn
)

report = sysid.default_report(
    models_sequences=models_sequences,
    initial_params=params,
    opt_params=opt_params,
    residual_fn=residual_fn,
    opt_result=opt_result,
    title="Labyrinth Game Optimal Parameters",
    generate_videos=False,
)

print(report.build())
report.save("report.html")


def main():
    pass
