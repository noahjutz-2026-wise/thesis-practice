import math
import time

from dynamixel_sdk import PacketHandler, PortHandler

port = PortHandler("/dev/ttyUSB0")
packet = PacketHandler(2.0)
port.openPort()
port.setBaudRate(1000000)

for dxl_id in (1, 2):
    packet.write1ByteTxRx(port, dxl_id, 64, 1)

t0 = time.time()
try:
    while True:
        val = int(100 * math.sin(2 * math.pi * (time.time() - t0)))
        for dxl_id in (1, 2):
            packet.write2ByteTxOnly(port, dxl_id, 102, val & 0xFFFF)
        time.sleep(0.01)
except KeyboardInterrupt:
    for dxl_id in (1, 2):
        packet.write2ByteTxOnly(port, dxl_id, 102, 0)
        packet.write1ByteTxRx(port, dxl_id, 64, 0)
    port.closePort()
