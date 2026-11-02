import zmq

IN = "ipc:///tmp/bus_in"
OUT = "ipc:///tmp/bus_out"


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.XSUB) as xsub, ctx.socket(zmq.XPUB) as xpub:
        xsub.bind(IN)
        xpub.bind(OUT)

        zmq.proxy(xsub, xpub)


if __name__ == "__main__":
    main()
