import itertools
import threading
import time
from datetime import datetime

import zmq


def subscriber(sub):
    print("sub")
    while True:
        x = sub.recv_pyobj()
        print(datetime.fromtimestamp(x).strftime("%H:%M:%S:%f")[:-3])
        time.sleep(1)


def publisher(pub):
    print("pub")
    for i in itertools.count():
        pub.send_pyobj(time.time())


def run_proxy(xsub, xpub):
    zmq.proxy(xsub, xpub)


def main():
    ctx = zmq.Context.instance()
    with (
        ctx.socket(zmq.SUB) as sub,
        ctx.socket(zmq.PUB) as pub,
        ctx.socket(zmq.XSUB) as xsub,
        ctx.socket(zmq.XPUB) as xpub,
    ):
        xpub.bind("inproc://xpub")
        xsub.bind("inproc://xsub")

        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.CONFLATE, 1)
        # pub.setsockopt(zmq.CONFLATE, 1)

        pub.connect("inproc://xsub")
        sub.connect("inproc://xpub")

        threading.Thread(target=run_proxy, args=(xsub, xpub)).start()
        threading.Thread(target=subscriber, args=(sub,)).start()

        publisher(pub)
