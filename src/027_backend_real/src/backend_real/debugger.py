import pickle

import cv2 as cv
import zmq

from backend_real.broker import OUT
from backend_real.nodes.vision import VISION_TOPIC_KEY


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, VISION_TOPIC_KEY)
    sub.connect(OUT)

    while True:
        _topic, payload = sub.recv_multipart()
        est = pickle.loads(payload)
        cv.imshow("frame", est["img"])
        cv.waitKey(1)


if __name__ == "__main__":
    main()
