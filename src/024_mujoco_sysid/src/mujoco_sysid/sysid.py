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


model = spec.compile()
data = mujoco.MjData(model)

params = sysid.ParameterDict()

for axis in ("x", "y"):
    params.add(
        sysid.Parameter(
            f"gainrpm_{axis}",
            nominal=0.05,
            min_value=0.0001,
            max_value=1,
            modifier=lambda model, p, a=axis: setattr(
                model.actuator(f"motor_{a}"),
                "gainprm",
                [p.value[0]] + list(model.actuator(f"motor_{a}").gainprm[1:]),
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"armature_{axis}",
            nominal=0.005,
            min_value=0.001,
            max_value=0.1,
            modifier=lambda model, p, a=axis: setattr(
                model.joint(f"joint_{a}"), "armature", float(p.value[0])
            ),
        )
    )
    params.add(
        sysid.Parameter(
            f"friction_{axis}",
            nominal=0.001,
            min_value=0.0,
            max_value=0.5,
            modifier=lambda model, p, a=axis: setattr(
                model.joint(f"joint_{a}"), "frictionloss", float(p.value[0])
            ),
        )
    )

initial_state = sysid.create_initial_state(model, data.qpos, data.qvel, data.act)

DATA_GLOB = "../027_backend_real/sysid_data/*.npz"
CONTROL_SCALE = 1000

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
    t = np.arange(len(d["sensor"])) / 50
    meas = d["sensor"].copy()
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
