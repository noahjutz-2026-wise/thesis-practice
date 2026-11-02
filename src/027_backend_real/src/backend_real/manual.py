import pickle

import zmq

from backend_real.broker import IN
from backend_real.network.topics import TOPIC_KEY_ACTUATOR


def main() -> None:
    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.connect(IN)
    try:
        while True:
            x, y = map(int, input("x y (0-1000): ").split())
            pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps((x, y))])
    except (KeyboardInterrupt, ValueError, EOFError):
        pass
    finally:
        pub.send_multipart([TOPIC_KEY_ACTUATOR, pickle.dumps((0, 0))])


if __name__ == "__main__":
    main()
