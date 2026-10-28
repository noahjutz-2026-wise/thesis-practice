from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path.home()
CONFIG_DIR = Path.cwd()


def main():
    img_bgr = cv2.imread(IMAGE_DIR.glob("*.jpg").__next__(), cv2.IMREAD_COLOR_BGR)
    if img_bgr is None:
        raise FileNotFoundError()

    markers_f = CONFIG_DIR.glob("*.csv").__next__()
    with open(markers_f, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    print(markers)


if __name__ == "__main__":
    main()
