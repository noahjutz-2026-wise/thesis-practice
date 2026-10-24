import os
import time

from dynamixel_sdk import PacketHandler, PortHandler

ACTUATOR_TOPIC_KEY = b"actuator"
DEV_MOTOR = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA2U4MF-if00-port0"
FREQUENCY = 50

ID_X = 1
ID_Y = 2

fd = os.timerfd_create(time.CLOCK_MONOTONIC)
os.timerfd_settime(fd, initial=1 / FREQUENCY, interval=1 / FREQUENCY)


def main():
    port = PortHandler(DEV_MOTOR)
    packet = PacketHandler(2.0)
    port.openPort()
    port.setBaudRate(1_000_000)

    for dxl_id in (1, 2):
        packet.write1ByteTxRx(port, dxl_id, 64, 1)

    try:
        while True:
            os.read(fd, 8)
            packet.write2ByteTxOnly(port, ID_X, 102, TODO & 0xFFFF)
            packet.write2ByteTxOnly(port, ID_Y, 102, TODO & 0xFFFF)
    except KeyboardInterrupt:
        for dxl_id in (1, 2):
            packet.write2ByteTxRx(port, dxl_id, 102, 0)
            packet.write1ByteTxRx(port, dxl_id, 64, 0)
        port.closePort()

    try:
        while True:
            val = int(100 * math.sin(2 * math.pi * (time.time() - t0)))
            for dxl_id in (1, 2):
                packet.write2ByteTxOnly(port, dxl_id, 102, val & 0xFFFF)
            time.sleep(0.01)
    except KeyboardInterrupt:
        for dxl_id in (1, 2):
            packet.write2ByteTxRx(port, dxl_id, 102, 0)
            packet.write1ByteTxRx(port, dxl_id, 64, 0)
        port.closePort()


if __name__ == "__main__":
    main()
