import sys

import cv2 as cv

DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


def main():
    cap = cv.VideoCapture(DEV_CAM)
    ret, frame = cap.read()
    if not ret:
        sys.exit(1)

    cv.imshow("frame", frame)
    while True:
        cv.waitKey(1)
