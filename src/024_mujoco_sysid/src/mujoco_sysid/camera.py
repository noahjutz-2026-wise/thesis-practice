import cv2 as cv
import numpy as np
import zmq

from mujoco_sysid.broker import IN, OUT

DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


def device(path: str) -> cv.VideoCapture:
    cap = cv.VideoCapture(path)
    return cap


def camera(device: cv.VideoCapture):
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.connect(IN)
    while True:
        pub.send_pyobj({"hi": "hello world"})


def main():
    camera(device(DEV_CAM))


if __name__ == "__main__":
    main()
