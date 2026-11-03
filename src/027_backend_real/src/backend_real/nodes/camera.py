import logging
import os
import pickle
from collections.abc import Callable

import cv2
import zmq

from backend_real.network.devices import get_device_camera
from backend_real.network.topics import TOPIC_KEY_CAMERA
from backend_real.nodes.broker import IN

logger = logging.getLogger(__name__)


def device(path: str) -> cv2.VideoCapture:
    real_path = os.path.realpath(path)
    cap = cv2.VideoCapture(real_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"camera not open: {real_path}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def camera(get_device: Callable[[], cv2.VideoCapture]):
    device = get_device()
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(IN)

    logger.info(f"up: {device.getBackendName()}")

    misses = 0
    while True:
        ret, frame = device.read()
        if not ret:
            misses += 1
            logger.warning(f"frame miss {misses}")
            if misses >= 60:
                raise RuntimeError("60 consecutive misses")
            continue
        misses = 0

        t = device.get(cv2.CAP_PROP_POS_MSEC)
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
