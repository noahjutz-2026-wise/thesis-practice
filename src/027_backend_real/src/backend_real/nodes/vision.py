import logging
import pickle

import zmq

from backend_real.cv.tracking import estimate_fn, open_markers
from backend_real.network.topics import TOPIC_KEY_CAMERA, TOPIC_KEY_VISION_TRACK_CORNERS
from backend_real.nodes.broker import IN, OUT

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_KEY_CAMERA)
        sub.connect(OUT)
        pub.connect(IN)

        current_estimate = open_markers()
        est = estimate_fn()

        logger.info("up")

        while True:
            _topic, payload = sub.recv_multipart()
            t, img = pickle.loads(payload)
            yaw, roll, pitch = est(img, current_estimate)
            pub.send_multipart([TOPIC_KEY_VISION_TRACK_CORNERS, pickle.dumps([])])


if __name__ == "__main__":
    main()
