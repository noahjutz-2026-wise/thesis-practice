import logging
import os
import pickle
from collections.abc import Callable

import cv2 as cv
import zmq

from backend_real.network.devices import get_device_camera
from backend_real.network.topics import TOPIC_KEY_CAMERA
from backend_real.nodes.broker import IN

logger = logging.getLogger(__name__)


def device(path: str) -> cv.VideoCapture:
    real_path = os.path.realpath(path)
    cap = cv.VideoCapture(real_path, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    return cap


def camera(get_device: Callable[[], cv.VideoCapture]):
    device = get_device()
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(IN)

    logger.info(f"up: {device.getBackendName()}")

    while True:
        ret, frame = device.read()
        t = device.get(cv.CAP_PROP_POS_MSEC)
        if not ret:
            logger.info("frame miss")
            device.release()
            device = get_device()
            continue
        pub.send_multipart([TOPIC_KEY_CAMERA, pickle.dumps([t, frame])])


def main():
    logging.basicConfig(level=logging.INFO)
    camera(lambda: device(get_device_camera()))


if __name__ == "__main__":
    main()
