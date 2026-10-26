import os
import pickle
from collections.abc import Callable

import cv2 as cv
import numpy as np
import zmq

from mujoco_sysid.broker import IN

CAM_TOPIC_KEY = b"cam"
DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


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

    while True:
        ret, frame = device.read()
        t = device.get(cv.CAP_PROP_POS_MSEC)
        if not ret:
            print("frame miss")
            device.release()
            device = get_device()
            continue
        pub.send_multipart([CAM_TOPIC_KEY, pickle.dumps([t, frame])])


def main():
    camera(lambda: device(DEV_CAM))


if __name__ == "__main__":
    main()
