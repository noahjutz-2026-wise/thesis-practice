import itertools
import pickle
import time

import cv2
import numpy as np
import zmq

from backend_real.cv.tracking import open_calibration
from backend_real.network.topics import TOPIC_KEY_CAMERA, TOPIC_KEY_MPU
from backend_real.util import drain

RCVTIMEO_MS = 500


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_CAMERA)
    sub.setsockopt(zmq.RCVTIMEO, RCVTIMEO_MS)
    sub.connect("tcp://127.0.0.1:5556")

    sub_mpu = ctx.socket(zmq.SUB)
    sub_mpu.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_MPU)
    sub_mpu.setsockopt(zmq.RCVTIMEO, RCVTIMEO_MS)
    sub_mpu.connect("tcp://127.0.0.1:5556")

    # est_pts = np.zeros(())  # TODO

    K, D, w, h = open_calibration()
    newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        newCameraMatrix,
        (1280, 720),
        cv2.CV_32FC1,
    )

    for i in itertools.count():
        # est_pts_undistort = est_pts.copy().reshape(-1, 1, 2)
        # est_pts_undistort = cv2.fisheye.undistortPoints(
        #     est_pts, K, D, R=np.eye(3), P=newCameraMatrix
        # ).squeeze()
        try:
            _topic, payload = sub.recv_multipart()
        except zmq.Again:
            print("camera timeout")
            continue
        t, img = pickle.loads(payload)

        try:
            _topic, payload = sub_mpu.recv_multipart()
        except zmq.Again:
            print("mpu timeout")
            continue

        if i % 10 == 0:
            print(pickle.loads(payload))

        drain(sub)
        drain(sub_mpu)

        frame = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        # polylines = [
        #     points.astype(np.int32).reshape(-1, 1, 2)
        #     for points in (est_pts_undistort[:4], est_pts_undistort[4:])
        # ]
        # cv2.polylines(frame, polylines, isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
