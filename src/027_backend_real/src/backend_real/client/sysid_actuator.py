import itertools
import math
import pickle
import random
import time

import zmq

from backend_real.network.topics import TOPIC_KEY_ACTUATOR

FREQUENCY = 50


def ramp(t: float) -> tuple[float, float]:
    return 1 - abs(t % 4 - 2) / 2, 0.5


def sine(t: float) -> tuple[float, float]:
    return math.sin(5 * t), math.sin(10 * t)


def square(t: float) -> tuple[float, float]:
    return 1 if t % 0.8 < 0.4 else -1, 1 if t % 0.4 < 0.2 else -1


def random_signal(t: float) -> tuple[float, float]:
    return random.uniform(-1, 1), random.uniform(-1, 1)


def reset(t) -> tuple[float, float]:
    if t < 50:
        return -100, -100
    if t < 60:
        return 100, 100
    return 0, 0


def send(pub: zmq.Socket, cmd: tuple[float, float]) -> None:
    pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps(cmd)])


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect("tcp://127.0.0.1:5555")
    time.sleep(1)

    fns = (ramp, sine, square, random_signal)
    scales = [(1000, 1000), (1000, -1000), (-500, 1000), (250, -250)]
    n = 500
    try:
        for fn, s in itertools.product(fns, scales):
            sx, sy = s
            print(fn.__name__, s)
            for t in range(70):
                send(pub, reset(t))
            time.sleep(1)

            for t in range(n):
                x, y = fn(t)
                x *= sx
                y *= sy
                send(pub, (x, y))
    except KeyboardInterrupt:
        send(pub, (0.0, 0.0))


if __name__ == "__main__":
    main()
