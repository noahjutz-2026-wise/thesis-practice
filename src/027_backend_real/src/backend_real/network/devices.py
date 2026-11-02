import os

# DEVICE_CAMERA = "/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0"
# DEVICE_MOTOR = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4MF-if00-port0"
DEVICE_CAMERA = os.environ["DEVICE_CAMERA"]
DEVICE_MOTOR = os.environ["DEVICE_MOTOR"]
