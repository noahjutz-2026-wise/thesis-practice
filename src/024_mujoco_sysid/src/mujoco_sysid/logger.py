import pickle

import numpy as np
import zmq

from mujoco_sysid.broker import OUT
from mujoco_sysid.estimator import ESTIMATOR_TOPIC_KEY

CONTROL_TOPIC_KEY = b"control"


def main():
    ctx = zmq.Context.instance()

    sub_ctrl = ctx.socket(zmq.SUB)
    sub_ctrl.setsockopt(zmq.SUBSCRIBE, CONTROL_TOPIC_KEY)
    sub_ctrl.connect(OUT)

    sub_est = ctx.socket(zmq.SUB)
    sub_est.setsockopt(zmq.SUBSCRIBE, ESTIMATOR_TOPIC_KEY)
    sub_est.connect(OUT)

    poller = zmq.Poller()
    poller.register(sub_ctrl, zmq.POLLIN)
    poller.register(sub_est, zmq.POLLIN)

    ctrl_buf, est_buf = [], []

    while True:
        events = dict(poller.poll())

        if sub_ctrl in events:
            _topic, payload = sub_ctrl.recv_multipart()
            t_ms, xy = pickle.loads(payload)
            ctrl_buf.append([t_ms, list(xy)])
            if len(ctrl_buf) >= 1000:
                np.savez("control", data=np.array(ctrl_buf, dtype=object))
                ctrl_buf.clear()

        if sub_est in events:
            _topic, payload = sub_est.recv_multipart()
            est = pickle.loads(payload)
            est_buf.append([est["t"] * 1e3, [est["alpha"], est["beta"]]])
            if len(est_buf) >= 1000:
                np.savez("estimator", data=np.array(est_buf, dtype=object))
                est_buf.clear()


if __name__ == "__main__":
    main()
