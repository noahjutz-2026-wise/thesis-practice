import logging

import zmq

IN = "ipc:///run/zmq/bus_in"
OUT = "ipc:///run/zmq/bus_out"

HOST_SUB = "tcp://*:5555"
HOST_PUB = "tcp://*:5556"

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.XSUB) as xsub, ctx.socket(zmq.XPUB) as xpub:
        xsub.bind(IN)
        xsub.bind(HOST_SUB)
        xpub.bind(OUT)
        xpub.bind(HOST_PUB)

        logger.info("up")

        zmq.proxy(xsub, xpub)


if __name__ == "__main__":
    main()
