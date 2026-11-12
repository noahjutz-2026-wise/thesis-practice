import itertools
import sys
import time

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VIDEO_CAP_PATH = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


def main():
    cap = cv2.VideoCapture(VIDEO_CAP_PATH)
    if not cap.isOpened():
        sys.exit(1)

    for i in itertools.count():
        if i == 1000:
            break
        ret, frame = cap.read()
        if not ret:
            print("frame error")
            sys.exit(2)

        t_c = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        t_p = time.monotonic()
        print(t_p - t_c)
