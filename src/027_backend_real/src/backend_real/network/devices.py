import os


def get_device_camera():
    return os.environ.get("DEVICE_CAMERA", default="/dev/video0")


def get_device_motor():
    return os.environ.get("DEVICE_MOTOR", default="/dev/ttyUSB0")
