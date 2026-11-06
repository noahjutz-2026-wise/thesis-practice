import itertools
import threading
import time

import zmq


def subscriber(sub):
    print("sub")
    while True:
        x = sub.recv_pyobj()
        print(x)
        time.sleep(1)


def publisher(pub):
    print("pub")
    for i in itertools.count():
        pub.send_pyobj(i)


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.SUB) as sub, ctx.socket(zmq.PUB) as pub:
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.bind("inproc://hello")
        pub.setsockopt(zmq.CONFLATE, 1)
        pub.connect("inproc://hello")

        threading.Thread(target=subscriber, args=(sub,)).start()

        publisher(pub)
