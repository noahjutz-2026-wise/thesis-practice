"""
Experiment control signal generator.

Publishes (x, y) actuator commands over ZMQ to the broker.

Strategy for high-entropy coverage with slow-speed bias:
  - Two sine waves with an irrational frequency ratio (1 : φ, the golden ratio)
    ensure the Lissajous trajectory never exactly repeats, eventually covering
    all (x, y) quadrant combinations.
  - A cube-root nonlinearity compresses the sine amplitude: most samples land
    near zero (slow speeds) while high-speed values are visited sparsely.
  - Each iteration runs for EXPERIMENT_DURATION seconds at PUBLISH_RATE Hz.
"""

import math
import pickle
import time

import zmq

from mujoco_sysid.actuator import ACTUATOR_TOPIC_KEY
from mujoco_sysid.broker import IN

PUBLISH_RATE = 50  # Hz
EXPERIMENT_DURATION = 120  # seconds per loop iteration

# Two sine periods (seconds).  Ratio ≈ golden ratio → irrational, non-repeating.
PERIOD_X = 13.0
PERIOD_Y = PERIOD_X / 1.6180339887  # ≈ 8.03 s

ACTION_MAX = 1000


def _shaped(phase: float) -> int:
    """Map a phase angle to an integer actuator value with slow-speed bias.

    sin gives a value in [-1, 1].  Applying the signed cube root compresses
    most probability mass near zero while preserving the sign and full range.
    """
    raw = math.sin(phase)  # [-1, 1], uniform over time
    shaped = math.copysign(abs(raw) ** (1 / 3), raw)  # bias toward 0
    return int(shaped * ACTION_MAX)


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect(IN)

    # Brief settle so the broker can register the connection
    time.sleep(0.1)

    iteration = 0
    try:
        while True:
            iteration += 1
            t_start = time.monotonic()
            t_end = t_start + EXPERIMENT_DURATION
            print(f"[main] iteration {iteration} — running for {EXPERIMENT_DURATION}s")

            t = 0.0
            while True:
                now = time.monotonic()
                if now >= t_end:
                    break

                x = _shaped(2 * math.pi * t / PERIOD_X)
                y = _shaped(2 * math.pi * t / PERIOD_Y)

                pub.send_multipart([ACTUATOR_TOPIC_KEY, pickle.dumps((x, y))])

                t += 1.0 / PUBLISH_RATE
                time.sleep(1.0 / PUBLISH_RATE)

    except KeyboardInterrupt:
        # Zero out on exit
        pub.send_multipart([ACTUATOR_TOPIC_KEY, pickle.dumps((0, 0))])
        print("[main] stopped.")


if __name__ == "__main__":
    main()
