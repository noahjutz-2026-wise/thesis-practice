from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"
IMAGE_DIR = Path.home()
CONFIG_DIR = Path.cwd()
HSV_BOUNDS = ((95, 177, 86), (112, 255, 162))
HSV_BOUNDS = ((49, 128, 108), (136, 255, 255))


def build_detector():
    d = 15

    params = cv2.SimpleBlobDetector.Params()
    params.filterByColor = False
    area = np.pi * (d / 2) ** 2
    params.filterByArea = True
    params.minArea = area * 0.5
    params.maxArea = area * 1.5
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.1

    return cv2.SimpleBlobDetector.create(params)


def preprocess(img: MatLike) -> MatLike:
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, *HSV_BOUNDS)
    return mask


def track(
    img: MatLike,
    estimate: NDArray,
    detect: Callable[[MatLike], Sequence[cv2.KeyPoint]],
) -> NDArray:

    subimgs = []
    for i, pos in enumerate(estimate):
        subimg = cv2.getRectSubPix(img, (100, 100), tuple(pos))
        keypoints = detect(subimg)
        if keypoints:
            pos2 = keypoints[0].pt
            cv2.circle(subimg, np.array(pos2).astype(int), 20, (255, 0, 0))
            estimate[i, :] = pos2 + pos - 50
        subimgs.append(subimg)

    cv2.imshow("", np.hstack(subimgs))
    cv2.waitKey(50)

    return estimate


def main():
    markers_f = CONFIG_DIR.glob("*.csv").__next__()
    with open(markers_f, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    scale_factor = np.array([1280 / 640, 720 / 400])
    markers *= scale_factor
    estimate = markers.copy()

    detector = build_detector()
    cap = cv2.VideoCapture(DEV_CAM)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("framer error")
            continue
        track(preprocess(frame), estimate, detector.detect)


if __name__ == "__main__":
    main()
