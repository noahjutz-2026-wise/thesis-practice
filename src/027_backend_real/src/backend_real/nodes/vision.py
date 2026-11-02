import pickle

import zmq

from backend_real.cv.tracking import estimate_fn, open_markers
from backend_real.network.topics import TOPIC_KEY_CAMERA, TOPIC_KEY_VISION
from backend_real.nodes.broker import IN, OUT


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_CAMERA)
        sub.connect(OUT)
        pub.connect(IN)

        current_estimate = open_markers()
        est = estimate_fn()

        while True:
            _topic, payload = sub.recv_multipart()
            t, img = pickle.loads(payload)
            yaw, roll, pitch = est(img, current_estimate)
            pub.send_multipart([TOPIC_KEY_VISION, pickle.dumps({yaw, roll, pitch})])


if __name__ == "__main__":
    main()
