import os
import sys
import time

import cv2 as cv
from dynamixel_sdk import PacketHandler, PortHandler

DEV_DYN = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4MF-if00-port0"
DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"
FREQUENCY = 50

fd = os.timerfd_create(time.CLOCK_MONOTONIC)
os.timerfd_settime(fd, initial=1 / FREQUENCY, interval=1 / FREQUENCY)


def main():
    cap = cv.VideoCapture(DEV_CAM)
    if not cap.isOpened():
        sys.exit(1)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("not ret")

        timestamp_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        now_ms = time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0
        latency_ms = now_ms - timestamp_ms

        print(f"Frame Latency: {latency_ms:.2f} ms")

        cv.imshow("frame", frame)
        cv.waitKey(1)

        print(frame)
        break

    # while True:
    #     os.read(fd, 8)
    #     print("hi")

    # port = PortHandler(DEV_DYN)
    # packet = PacketHandler(2.0)
    # port.openPort()
    # port.setBaudRate(1_000_000)

    # for dxl_id in (1, 2):
    #     packet.write1ByteTxRx(port, dxl_id, 64, 1)

    # t0 = time.time()
    # try:
    #     while True:
    #         val = int(100 * math.sin(2 * math.pi * (time.time() - t0)))
    #         for dxl_id in (1, 2):
    #             packet.write2ByteTxOnly(port, dxl_id, 102, val & 0xFFFF)
    #         time.sleep(0.01)
    # except KeyboardInterrupt:
    #     for dxl_id in (1, 2):
    #         packet.write2ByteTxRx(port, dxl_id, 102, 0)
    #         packet.write1ByteTxRx(port, dxl_id, 64, 0)
    #     port.closePort()
