import pickle

import cv2 as cv
import zmq

from backend_real.network.topics import TOPIC_KEY_CAMERA, TOPIC_KEY_VISION


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_CAMERA)
    sub.connect("tcp://127.0.0.1:5556")

    while True:
        _topic, payload = sub.recv_multipart()
        t, img = pickle.loads(payload)
        print(t)
        # cv.imshow("frame", img)
        # cv.waitKey(1)


if __name__ == "__main__":
    main()
