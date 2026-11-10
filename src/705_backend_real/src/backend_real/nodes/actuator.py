import logging
import os
import pickle
import time

import zmq
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from backend_real.network.devices import get_device_motor
from backend_real.network.topics import TOPIC_KEY_ACTUATOR
from backend_real.nodes.broker import OUT

logger = logging.getLogger(__name__)

FREQUENCY = 50

ID_X = 1
ID_Y = 2

fd = os.timerfd_create(time.CLOCK_MONOTONIC)
os.timerfd_settime(fd, initial=1 / FREQUENCY, interval=1 / FREQUENCY)


def main():
    logging.basicConfig(level=logging.INFO)
    port = PortHandler(get_device_motor())
    packet = PacketHandler(2.0)

    if not port.openPort():
        raise SystemExit("openPort failed")
    if not port.setBaudRate(1_000_000):
        raise SystemExit("setBaudRate failed")

    for dxl_id in (1, 2):
        res, _ = packet.write1ByteTxRx(port, dxl_id, 64, 1)
        if res != COMM_SUCCESS:
            raise RuntimeError("Torque on failed")

    sub_act = zmq.Context.instance().socket(zmq.SUB)
    sub_act.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_ACTUATOR)
    sub_act.connect(OUT)

    logger.info(f"up: {port.is_open}")

    try:
        while True:
            os.read(fd, 8)

            _topic, payload = sub_act.recv_multipart()
            x, y = pickle.loads(payload)

            res = packet.write2ByteTxOnly(port, ID_X, 102, int(x) & 0xFFFF)
            if res != COMM_SUCCESS:
                raise RuntimeError(f"ID_X: {res}")
            res = packet.write2ByteTxOnly(port, ID_Y, 102, int(y) & 0xFFFF)
            if res != COMM_SUCCESS:
                raise RuntimeError(f"ID_Y: {res}")
    except KeyboardInterrupt:
        for dxl_id in (1, 2):
            packet.write2ByteTxRx(port, dxl_id, 102, 0)
            packet.write1ByteTxRx(port, dxl_id, 64, 0)
        port.closePort()


if __name__ == "__main__":
    main()
