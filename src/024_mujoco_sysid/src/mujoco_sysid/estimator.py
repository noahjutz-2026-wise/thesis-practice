import pickle
from pathlib import Path

import zmq
from crs2r_image_processing import EstimationPipeline, generate_default_markers

from mujoco_sysid.broker import IN, OUT
from mujoco_sysid.camera import CAM_TOPIC_KEY

ESTIMATOR_TOPIC_KEY = b"estimator"


def main():
    pipeline = EstimationPipeline(
        fps=55.0, markers=Path("~/Downloads/markers.csv").expanduser()
    )
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.SUBSCRIBE, CAM_TOPIC_KEY)
        sub.connect(OUT)
        pub.connect(IN)

        while True:
            _topic, payload = sub.recv_multipart()
            t, img = pickle.loads(payload)
            x_hat, _P, inputs, _xb, _yb = pipeline.estimate(img)
            dic = {
                "img": pipeline.measurements.annotate(img),
                "x_b": x_hat[0],
                "y_b": x_hat[1],
                "vx_b": x_hat[2],
                "vy_b": x_hat[3],
                "alpha": inputs[0],
                "beta": inputs[1],
                "t": t,
            }
            pub.send_multipart([ESTIMATOR_TOPIC_KEY, pickle.dumps(dic)])


if __name__ == "__main__":
    main()
