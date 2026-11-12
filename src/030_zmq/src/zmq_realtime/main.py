import itertools
import threading
import time
from datetime import datetime

import zmq


def subscriber(sub):
    print("sub")
    while True:
        id, t = sub.recv_pyobj()
        print(id, datetime.fromtimestamp(t).strftime("%H:%M:%S:%f")[:-3])
        time.sleep(1)


def time_publisher(pub, id=-1):
    print("pub time")
    for i in itertools.count():
        pub.send_pyobj([id, time.time()])


def run_proxy(xsub, xpub):
    zmq.proxy(xsub, xpub)


def main():
    ctx = zmq.Context.instance()

    xsub = ctx.socket(zmq.XSUB)
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind("inproc://xpub")
    xsub.bind("inproc://xsub")

    subs = []
    pubs = []

    for i in range(50):
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect("inproc://xpub")
        subs.append(sub)

        pub = ctx.socket(zmq.PUB)
        pub.connect("inproc://xsub")
        pubs.append(pub)

    threading.Thread(target=run_proxy, args=(xsub, xpub)).start()
    for i, pub in enumerate(pubs[1:]):
        threading.Thread(target=time_publisher, args=(pub,), kwargs={"id": i}).start()

    threading.Thread(target=time_publisher, args=(pubs[0],)).start()

    subscriber(subs[0])
