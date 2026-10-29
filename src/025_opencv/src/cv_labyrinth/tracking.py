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

OUTER_X = 0.330
OUTER_Y = 0.286
INNER_X = 0.290
INNER_Y = 0.245
WALL_THICKNESS = 0.0075
DOT_OUTER_DY = 0.050 + WALL_THICKNESS
DOT_DIAMETER = 0.008


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


def build_rect(dx, dy) -> NDArray:
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    objp = signs * [dx, dy]
    objp = np.pad(objp, ((0, 0), (0, 1)))
    objp = np.ascontiguousarray(objp)
    return objp


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
    for i, pos in enumerate(estimate):
        subimg = cv2.getRectSubPix(img, (100, 100), tuple(pos))
        if (cv2.countNonZero(subimg) / subimg.size) > 0.05:
            print("flash frame")
            continue
        keypoints = detect(subimg)
        if keypoints:
            pos2 = keypoints[0].pt
            estimate[i, :] = pos2 + pos - 50
    return estimate


def pose_fn(object_points: NDArray, K: NDArray, D: NDArray):
    object_points = object_points.reshape(-1, 1, 3)

    def pose(image_points: NDArray) -> None:
        image_points = image_points.reshape(-1, 1, 2)

        ret, rvec, _tvec = cv2.fisheye.solvePnP(
            object_points, image_points, K, D, flags=cv2.SOLVEPNP_IPPE
        )

        return rvec

    return pose


def main():
    calib = cv2.FileStorage(CONFIG_DIR / "calibration.yaml", cv2.FILE_STORAGE_READ)
    K = calib.getNode("camera_matrix").mat()
    D = calib.getNode("distortion_coefficients").mat()
    w = calib.getNode("image_width").real()
    w = int(w)
    h = calib.getNode("image_height").real()
    h = int(h)
    calib.release()
    newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        newCameraMatrix,
        (1280, 720),
        cv2.CV_32FC1,
    )

    markers_f = CONFIG_DIR.glob("*.csv").__next__()
    with open(markers_f, "r") as f:
        data = f.read()
        markers = np.fromstring(data.replace(",", " "), sep=" ").reshape(-1, 2)

    scale_factor = np.array([1280 / 640, 720 / 400])
    markers *= scale_factor
    estimate = markers.copy()

    detector = build_detector()
    cap = cv2.VideoCapture(DEV_CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    objp_inner = build_rect(
        dx=INNER_X / 2 - DOT_DIAMETER, dy=INNER_Y / 2 - DOT_DIAMETER
    )
    objp_outer = build_rect(
        dx=OUTER_X / 2 - DOT_DIAMETER, dy=OUTER_Y / 2 - DOT_OUTER_DY - DOT_DIAMETER
    )

    pose_inner = pose_fn(objp_inner, K, D)
    pose_outer = pose_fn(objp_outer, K, D)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame error")
            continue
        track(preprocess(frame), estimate, detector.detect)

        # PnP
        print("==")
        r_inner = pose_inner(estimate[:4])
        print(r_inner)
        r_outer = pose_outer(estimate[4:])
        print(r_outer)

        est_undistorted = estimate.copy().reshape(-1, 1, 2)
        est_undistorted = cv2.fisheye.undistortPoints(
            est_undistorted, K, D, R=np.eye(3), P=newCameraMatrix
        ).squeeze()

        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        for pos in est_undistorted:
            cv2.circle(frame, pos.astype(int), 20, (0, 255, 0))
        cv2.imshow("f", frame)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()
