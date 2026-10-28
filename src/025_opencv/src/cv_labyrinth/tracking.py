from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

IMAGE_DIR = Path.home()
CONFIG_DIR = Path.cwd()
HSV_BOUNDS = ((95, 177, 86), (112, 255, 162))


def build_detector():
    d = 10

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
    markers: NDArray,
    estimate: NDArray,
    detect: Callable[[MatLike], Sequence[cv2.KeyPoint]],
) -> NDArray:

    subimgs = []
    for i, pos in enumerate(estimate):
        subimg = cv2.getRectSubPix(img, (100, 100), tuple(pos))
        keypoints = detect(subimg)
        if keypoints:
            pos2 = keypoints[0].pt
            print(pos - pos2)
            cv2.circle(subimg, np.array(pos2).astype(int), 20, (255, 0, 0))
        else:
            cv2.circle(subimg, (50, 50), 10, (255, 0, 0))
            print(f".{pos}")
        subimgs.append(subimg)

    cv2.imshow("", np.hstack(subimgs))
    while True:
        cv2.waitKey(1)

    return np.zeros(())


def main():
    markers_f = CONFIG_DIR.glob("*.csv").__next__()
    with open(markers_f, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    scale_factor = np.array([1280 / 640, 720 / 400])
    scaled_points = markers * scale_factor

    img = cv2.imread(IMAGE_DIR.glob("*.jpg").__next__(), cv2.IMREAD_COLOR_BGR)
    if img is None:
        raise FileNotFoundError()

    detector = build_detector()

    track(preprocess(img), scaled_points, scaled_points, detector.detect)


if __name__ == "__main__":
    main()
