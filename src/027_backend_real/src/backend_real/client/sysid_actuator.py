import itertools
import os
import pickle
import random
import time

import zmq

from backend_real.network.topics import TOPIC_KEY_ACTUATOR

FREQUENCY = 50


def random_signal(t: float) -> tuple[float, float]:
    return random.uniform(-1, 1), random.uniform(-1, 1)


def reset(t) -> tuple[float, float]:
    if t < 10:
        return -100, -100
    if t < 15:
        return 100, 100
    return 0, 0


def send(pub: zmq.Socket, cmd: tuple[float, float]) -> None:
    pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps(cmd)])


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect("tcp://127.0.0.1:5555")
    time.sleep(1)
    fd = os.timerfd_create(time.CLOCK_MONOTONIC)
    os.timerfd_settime(fd, initial=1 / FREQUENCY, interval=1 / FREQUENCY)

    fns = (random_signal,)
    scales = [(50, -50), (100, -100), (200, -200)]
    n = 800
    try:
        for fn, s in itertools.product(fns, scales):
            sx, sy = s
            print(fn.__name__, s)
            for t in range(70):
                os.read(fd, 8)
                send(pub, reset(t))
            time.sleep(1)

            for t in range(n):
                os.read(fd, 8)
                x, y = fn(t * max(sx, sy) // 10)
                x *= sx
                y *= sy
                print(x, y)
                send(pub, (x, y))
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        send(pub, (0.0, 0.0))


if __name__ == "__main__":
    main()
