import pickle
from pathlib import Path

import zmq

from backend_real.broker import IN, OUT
from backend_real.network.topics import TOPIC_KEY_CAMERA, TOPIC_KEY_VISION


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_CAMERA)
        sub.connect(OUT)
        pub.connect(IN)

        while True:
            _topic, payload = sub.recv_multipart()
            t, img = pickle.loads(payload)
            # TODO run opencv estimator
            pub.send_multipart([TOPIC_KEY_VISION, pickle.dumps({})])


if __name__ == "__main__":
    main()
