from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

IMAGE_DIR = Path.home()
CONFIG_DIR = Path.cwd()


def track(img: MatLike, markers: np.ndarray):
    for marker in markers:
        print(marker)
        cv2.circle(img, tuple(marker.astype(int)), 20, (255, 0, 0))

    cv2.imshow("frame", img)
    height, width = img.shape[:2]
    print(f"{width}x{height}")
    while True:
        cv2.waitKey(1)


def main():
    img_bgr = cv2.imread(IMAGE_DIR.glob("*.jpg").__next__(), cv2.IMREAD_COLOR_BGR)
    if img_bgr is None:
        raise FileNotFoundError()
    markers_f = CONFIG_DIR.glob("*.csv").__next__()
    with open(markers_f, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    scale_factor = np.array([1280 / 640, 720 / 400])
    scaled_points = markers * scale_factor
    track(img_bgr, scaled_points)


if __name__ == "__main__":
    main()
