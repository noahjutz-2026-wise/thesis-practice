import logging
import pickle
import time
from collections.abc import Callable

import numpy as np
import serial
import zmq

from backend_real.network.devices import get_device_mpu
from backend_real.network.topics import TOPIC_KEY_MPU
from backend_real.nodes.broker import IN

logger = logging.getLogger(__name__)

BAUDRATE = 230_400
TIMEOUT_S = 0.5


def device(path: str) -> tuple[serial.Serial, float]:
    s = serial.Serial(path, BAUDRATE, timeout=TIMEOUT_S)
    mcu_us = None
    while mcu_us is None:
        line = s.readline()
        try:
            mcu_us = int(line.split(b",")[0])
        except ValueError:
            continue
    offset = time.monotonic() * 1e6 - mcu_us
    return s, offset


def mpu(get_device: Callable[..., tuple[serial.Serial, float]]):
    s, offset = get_device()
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(IN)

    logger.info(f"up: {s.port}")

    while True:
        line = s.readline()
        if not line:
            raise RuntimeError(f"mpu: no serial data for {TIMEOUT_S} s")
        row = np.array(line.split(b","), dtype=np.float64)
        row[0] = (row[0] + offset) / 1e6  # mcu micros -> host monotonic seconds
        pub.send_multipart([TOPIC_KEY_MPU, pickle.dumps(row)])


def main():
    logging.basicConfig(level=logging.INFO)
    mpu(lambda: device(get_device_mpu()))


if __name__ == "__main__":
    main()
