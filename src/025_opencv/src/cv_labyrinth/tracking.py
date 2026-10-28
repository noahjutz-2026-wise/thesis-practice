from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

IMAGE_DIR = Path.home()
CONFIG_DIR = Path.cwd()


def track(img: MatLike, markers: NDArray, estimate: NDArray) -> NDArray:
    img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
    for marker in markers:
        print(marker)
        cv2.circle(img, tuple(marker.astype(int)), 20, (255, 0, 0))

    subimgs = []
    for i, pos in enumerate(estimate):
        x, y = pos.astype(int)
        subimg = cv2.getRectSubPix(img, (100, 100), tuple(pos))
        subimgs.append(subimg)
        print(pos)

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

    img_bgr = cv2.imread(IMAGE_DIR.glob("*.jpg").__next__(), cv2.IMREAD_COLOR_BGR)
    if img_bgr is None:
        raise FileNotFoundError()

    track(img_bgr, scaled_points, scaled_points)


if __name__ == "__main__":
    main()
