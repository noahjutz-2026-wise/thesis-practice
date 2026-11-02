import pickle
from pathlib import Path

import zmq

from backend_real.broker import IN, OUT

from .camera import CAM_TOPIC_KEY

VISION_TOPIC_KEY = b"estimator"


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.SUBSCRIBE, CAM_TOPIC_KEY)
        sub.connect(OUT)
        pub.connect(IN)

        while True:
            _topic, payload = sub.recv_multipart()
            t, img = pickle.loads(payload)
            # TODO run opencv estimator
            pub.send_multipart([VISION_TOPIC_KEY, pickle.dumps({})])


if __name__ == "__main__":
    main()
