import cv2 as cv
import zmq

from mujoco_sysid.broker import OUT


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(OUT)

    while True:
        t, img = sub.recv_pyobj()
        print(t)
        cv.imshow("frame", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
