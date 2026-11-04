import os
import pickle
import time

import numpy as np
import zmq

from backend_real.network import topics

FREQUENCY = 200
OUT_DIR = "sysid_data"
TIMEOUT = 0.2
TRIM = 300


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, topics.TOPIC_KEY_MPU)
    sub.setsockopt(zmq.SUBSCRIBE, topics.TOPIC_KEY_ACTUATOR)
    sub.connect("tcp://127.0.0.1:5556")
    fd = os.timerfd_create(time.CLOCK_MONOTONIC)
    os.timerfd_settime(fd, initial=1 / FREQUENCY, interval=1 / FREQUENCY)

    last_act_t = -1.0
    sensors = []
    actions = []
    sensor = None
    act = None
    while True:
        os.read(fd, 8)
        now = time.time()
        while True:
            try:
                topic, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            match topic:
                case topics.TOPIC_KEY_ACTUATOR:
                    act = np.array(pickle.loads(payload), dtype=np.float64)
                    last_act_t = now
                case topics.TOPIC_KEY_MPU:
                    sensor = pickle.loads(payload)[1:7]

        if now - last_act_t >= TIMEOUT:
            if len(sensors) > 2 * TRIM:
                np.savez(
                    f"{OUT_DIR}/batch_{int(now)}.npz",
                    sensor=np.array(sensors[TRIM:-TRIM]),
                    action=np.array(actions[TRIM:-TRIM]),
                )
            sensors = []
            actions = []
            continue
        sensors.append(sensor)
        actions.append(act)


if __name__ == "__main__":
    main()
