import math
import pickle
import random
import time
from collections.abc import Iterator
from itertools import count, islice

import zmq

from backend_real.network.topics import TOPIC_KEY_ACTUATOR

FREQUENCY = 50
N_BASE = 10
SCALES = [(1000, 1000), (1000, -1000), (-500, 1000), (250, -250)]


def ramp() -> Iterator[tuple[float, float]]:
    return ((1 - abs(t % 200 - 100) / 100, 0.5) for t in count())


def sine() -> Iterator[tuple[float, float]]:
    return ((math.sin(t / 10), math.sin(t / 5)) for t in count())


def square() -> Iterator[tuple[float, float]]:
    return ((t // 20 % 2 * 2 - 1, t // 10 % 2 * 2 - 1) for t in count())


def random_signal() -> Iterator[tuple[float, float]]:
    return ((random.uniform(-1, 1), random.uniform(-1, 1)) for _ in count())


def variation(x: float, y: float, scalex: float, scaley: float) -> tuple[float, float]:
    return x * scalex, y * scaley


def reset() -> Iterator[tuple[float, float]]:
    return ((-100, -100) for _ in range(50))


def send(pub: zmq.Socket, cmd: tuple[float, float]) -> None:
    pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps(cmd)])


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect("tcp://127.0.0.1:5555")
    time.sleep(1)
    for fn in (ramp, sine, square, random_signal):
        for sx, sy in SCALES:
            gen = fn()
            n = round(N_BASE * max(map(abs, (sx, sy))) / min(map(abs, (sx, sy))))
            for _ in reset():
                send(pub, (100, 100))
                time.sleep(1 / FREQUENCY)
            send(pub, (0, 0))
            time.sleep(1)
            for x, y in islice(gen, n):
                send(pub, variation(x, y, sx, sy))
                time.sleep(1 / FREQUENCY)
            send(pub, (0, 0))
            time.sleep(1)
    send(pub, (0, 0))


if __name__ == "__main__":
    main()
