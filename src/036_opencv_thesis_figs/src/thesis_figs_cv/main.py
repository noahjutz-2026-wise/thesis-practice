from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike

IMAGE_FILE = Path.home() / "cyberrunner_top.jpg"
OUT_PATH = Path(__file__).parents[2] / "outputs"

HSV_BOUNDS = ((49, 128, 108), (136, 255, 255))
FILE_CAM_MARKERS = Path(__file__).parents[3] / "025_opencv/markers.csv"
FILE_CAM_CONFIG = Path(__file__).parents[3] / "025_opencv/calibration.yaml"


def open_markers():
    with open(FILE_CAM_MARKERS, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    scale_factor = np.array([1280 / 640, 720 / 400])
    markers *= scale_factor
    return markers


def preprocess(f: MatLike) -> MatLike:
    f = cv2.GaussianBlur(f, (5, 5), 0)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(f, *HSV_BOUNDS)
    return mask


def subimg(f: MatLike, pos: npt.NDArray) -> MatLike:
    return cv2.getRectSubPix(f, (100, 100), tuple(pos))


def main():
    im = cv2.imread(IMAGE_FILE)
    assert im is not None
    markers = open_markers()

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    prep = preprocess(im)
    cv2.imwrite(OUT_PATH / "prep.jpg", prep)

    subs = [subimg(prep, pos) for pos in markers]
    for i, sub in enumerate(subs):
        cv2.imwrite(OUT_PATH / f"sub{i}.jpg", sub)


if __name__ == "__main__":
    main()
