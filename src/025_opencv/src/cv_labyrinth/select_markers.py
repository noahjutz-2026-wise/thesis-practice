"""Click 8 markers in order on the live DEV_CAM feed; save as CSV (x,y per line)."""

import cv2 as cv
import numpy as np

from cv_labyrinth.blobs import DEV_CAM

N = 8
pts = []


def main():
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == N:
                np.savetxt("markers.csv", pts, fmt="%d", delimiter=",")

    cv.namedWindow("select", cv.WINDOW_NORMAL)
    cv.resizeWindow("select", 640, 400)
    cv.setMouseCallback("select", on_mouse)
    cap = cv.VideoCapture(DEV_CAM)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.resize(frame, (640, 400))
        for p in pts:
            cv.circle(frame, p, 4, (0, 0, 255), -1)
        cv.imshow("select", frame)
        if cv.waitKey(1) == 27:
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
