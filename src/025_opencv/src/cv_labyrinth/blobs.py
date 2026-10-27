import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path("~").expanduser()
DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


def main():
    img_bgr = cv2.imread(IMAGE_DIR.glob("*.jpg").__next__(), cv2.IMREAD_COLOR_BGR)
    if img_bgr is None:
        raise FileNotFoundError()

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img = cv2.GaussianBlur(img, (11, 11), 0)

    mask1 = cv2.inRange(img, (95, 177, 86), (112, 255, 162))

    im = cv2.bitwise_and(img_bgr, img_bgr, mask=mask1)

    params = cv2.SimpleBlobDetector.Params()
    params.filterByColor = False

    # Filter by Area
    d = 10
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

    detector = cv2.SimpleBlobDetector.create(params)
    keypoints = detector.detect(im)
    if not keypoints:
        print("no keypoints")
        sys.exit()
    output = cv2.drawKeypoints(
        img_bgr,
        keypoints,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imshow("Blobs Detected", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
