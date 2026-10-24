import pickle

import cv2 as cv
import zmq

from mujoco_sysid.broker import OUT
from mujoco_sysid.camera import CAM_TOPIC_KEY


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, CAM_TOPIC_KEY)
    sub.connect(OUT)

    while True:
        topic, payload = sub.recv_multipart()
        t, img = pickle.loads(payload)
        print(t)
        cv.imshow("frame", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
