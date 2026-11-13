import itertools
import threading
import time
from datetime import datetime

import zmq

XPUB_ADDR_INPROC = "inproc://xpub"
XSUB_ADDR_INPROC = "inproc://xsub"

XPUB_ADDR_IPC = "ipc://ipc_xpub"
XSUB_ADDR_IPC = "ipc://ipc_xsub"

XPUB_ADDR = XPUB_ADDR_IPC
XSUB_ADDR = XSUB_ADDR_IPC


def len_subscriber(sub):
    print("len sub")
    while True:
        outs = []
        while True:
            try:
                r = sub.recv_pyobj(zmq.EAGAIN)
                outs.append(r)
            except zmq.Again:
                break
        print(len(outs))
        time.sleep(1)


def time_subscriber(sub):
    print("time sub")
    while True:
        id, t = sub.recv_pyobj()
        t2 = time.time()
        print(id, t2 - t)
        time.sleep(1)


def num_subscriber(sub):
    print("num sub")
    while True:
        id, i = sub.recv_pyobj()
        print(id, i)
        time.sleep(1)


def time_publisher(pub, id=-1):
    print("pub time")
    for i in itertools.count():
        pub.send_pyobj([id, time.time()])
        # time.sleep(0.0001)


def num_publisher(pub, id=-1):
    print("pub num")
    for i in itertools.count():
        pub.send_pyobj([id, i])
        # time.sleep(0.0001)


def run_proxy(xsub, xpub):
    zmq.proxy(xsub, xpub)


def main():
    ctx = zmq.Context.instance()

    xsub = ctx.socket(zmq.XSUB)
    # xsub.setsockopt(zmq.RCVHWM, 1)
    xpub = ctx.socket(zmq.XPUB)
    xpub.setsockopt(zmq.SNDHWM, 1)
    xpub.bind(XPUB_ADDR)
    xsub.bind(XSUB_ADDR)

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    # sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.connect(XPUB_ADDR)

    pubs = []

    for i in range(100):
        pub = ctx.socket(zmq.PUB)
        # pub.setsockopt(zmq.SNDHWM, 1)
        pub.connect(XSUB_ADDR)
        pubs.append(pub)

    threading.Thread(target=run_proxy, args=(xsub, xpub)).start()
    for i, pub in enumerate(pubs):
        threading.Thread(target=time_publisher, args=(pub,), kwargs={"id": i}).start()

    len_subscriber(sub)


if __name__ == "__main__":
    main()
