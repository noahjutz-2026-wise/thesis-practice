import pickle

import cv2 as cv
import zmq

from mujoco_sysid.broker import OUT
from mujoco_sysid.camera import CAM_TOPIC_KEY
from mujoco_sysid.estimator import ESTIMATOR_TOPIC_KEY


def main():
    ctx = zmq.Context.instance()
    sub_cam = ctx.socket(zmq.SUB)
    sub_cam.setsockopt(zmq.SUBSCRIBE, CAM_TOPIC_KEY)
    sub_cam.connect(OUT)

    sub_est = ctx.socket(zmq.SUB)
    sub_est.setsockopt(zmq.SUBSCRIBE, ESTIMATOR_TOPIC_KEY)
    sub_est.connect(OUT)

    while True:
        _topic, payload = sub_cam.recv_multipart()
        t, img = pickle.loads(payload)
        cv.imshow("frame", img)
        cv.waitKey(1)

        _topic, payload = sub_est.recv_multipart()
        est = pickle.loads(payload)
        print(est)


if __name__ == "__main__":
    main()
