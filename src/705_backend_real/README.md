# Backend Real

## Prerequisites

- Calibrate camera fisheye and select markers: [025_opencv](../025_opencv/README.md)
- Setup an arduino for MPU-6050: [028_mpu6050](../028_mpu6050/README.md)
- Pass camera, dynamixel, mpu devices in `compose.yml`

Default `compose.yml` configuration:

| Device                | Path                                                                      |
| --------------------- | ------------------------------------------------------------------------- |
| Camera                | `/dev/v4l/by-id/usb-e-con_systems_See3CAM_24CUG_2509460D-video-index0`    |
| Dynamixel U2D2        | `/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4MF-if00-port0` |
| MPU-6050 Arduino Nano | `/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A10N2HR3-if00-port0`          |

## Quickstart

```bash
# Start server
podman-compose up --build

# Interact
FILE_MARKERS_CSV=markers.csv \
    FILE_CAM_CONFIG_YAML=calibration.yaml \
    uv run python -m backend_real.client.debugger
uv run python -m backend_real.client.manual
```

## Development

```bash
uv sync --no-dev --group container # opencv-python-headless
uv sync # opencv-python
```
