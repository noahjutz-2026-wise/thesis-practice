import ctypes

import cv2 as cv
import iceoryx2 as iox2
import numpy as np

DEV_CAM = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"


def device(path: str) -> cv.VideoCapture:
    cap = cv.VideoCapture(path)
    return cap


def camera(device: cv.VideoCapture):
    node = (
        iox2.NodeBuilder.new()
        .name(iox2.NodeName.new("Camera"))
        .create(iox2.ServiceType.Ipc)
    )
    service = (
        node.service_builder(iox2.ServiceName.new("CamService"))
        .publish_subscribe(ctypes.c_float)
        .open_or_create()
    )
    publisher = service.publisher_builder().create()
    cap = device

    while True:
        node.wait(iox2.Duration.from_millis(100))
        sample = publisher.loan_uninit()
        sample = sample.write_payload(ctypes.c_float(np.random.rand()))
        sample.send()


def main():
    camera(device(DEV_CAM))


if __name__ == "__main__":
    main()
