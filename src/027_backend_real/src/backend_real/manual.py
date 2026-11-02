import pickle

import zmq

from backend_real.broker import IN
from backend_real.nodes.actuator import ACTUATOR_TOPIC_KEY


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect(IN)
    try:
        while True:
            x, y = map(int, input("x y (0-1000): ").split())
            pub.send_multipart([ACTUATOR_TOPIC_KEY, pickle.dumps((x, y))])
    except (KeyboardInterrupt, ValueError, EOFError):
        pass
    finally:
        pub.send_multipart([ACTUATOR_TOPIC_KEY, pickle.dumps((0, 0))])


if __name__ == "__main__":
    main()
