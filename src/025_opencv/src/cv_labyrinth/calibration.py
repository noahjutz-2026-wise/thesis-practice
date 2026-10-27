from pathlib import Path

import cv2 as cv
import numpy as np

IMAGE_DIR = Path("~/Pictures/calibration").expanduser()
CHECKERBOARD = (8, 6)
DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"
OUT_CONFIG = "calibration.yaml"


def main():
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    imgs = [img for img in IMAGE_DIR.glob("*.jpg")]
    objpoints = []
    imgpoints = []
    for im in imgs:
        img = cv.imread(im, cv.IMREAD_GRAYSCALE)

        found, corners = cv.findChessboardCorners(img, CHECKERBOARD, None)
        if not found:
            print(f"error with {im}")
            continue

        objpoints.append(objp)
        imgpoints.append(corners.reshape(1, -1, 2))
        assert objp.shape == (1, 48, 3) and corners.shape in (
            (48, 1, 2),
            (48, 2),
        ), f"error with {im}"

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    rms, K, D, rvecs, tvecs = cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        img.shape[::-1],
        K,
        D,
    )

    img = cv.imread(imgs[0])
    h, w = img.shape[:2]
    # img_size = (w, h)
    # map1, map2 = cv.fisheye.initUndistortRectifyMap(
    #     K, D, np.eye(3, dtype=np.float64), K, img_size, cv.CV_32FC1
    # )

    fs = cv.FileStorage(OUT_CONFIG, cv.FILE_STORAGE_WRITE)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", D)
    fs.release()


if __name__ == "__main__":
    main()
