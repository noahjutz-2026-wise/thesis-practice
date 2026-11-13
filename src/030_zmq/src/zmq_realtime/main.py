import itertools
import threading
import time

import zmq


def subscriber(sub):
    print("sub")
    while True:
        print("===")
        out = []
        while True:
            try:
                id, i = sub.recv_pyobj(zmq.NOBLOCK)
                out.append((id, i))
            except zmq.Again:
                break
        print(out)
        time.sleep(1)


def time_publisher(pub, id=-1):
    print("pub time")
    for i in itertools.count():
        pub.send_pyobj([id, time.time()])
        time.sleep(0.0001)


def num_publisher(pub, id=-1):
    print("pub num")
    for i in itertools.count():
        pub.send_pyobj([id, i])
        time.sleep(0.0001)


def run_proxy(xsub, xpub):
    zmq.proxy(xsub, xpub)


def main():
    ctx = zmq.Context.instance()

    xsub = ctx.socket(zmq.XSUB)
    xsub.setsockopt(zmq.RCVHWM, 1)
    xpub = ctx.socket(zmq.XPUB)
    xpub.setsockopt(zmq.SNDHWM, 1)
    xpub.bind("inproc://xpub")
    xsub.bind("inproc://xsub")

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    # sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.connect("inproc://xpub")

    pubs = []

    for i in range(1):
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 1)
        pub.connect("inproc://xsub")
        pubs.append(pub)

    threading.Thread(target=run_proxy, args=(xsub, xpub)).start()
    for i, pub in enumerate(pubs):
        threading.Thread(target=num_publisher, args=(pub,), kwargs={"id": i}).start()

    subscriber(sub)


if __name__ == "__main__":
    main()
