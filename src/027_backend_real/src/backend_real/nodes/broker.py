import zmq

IN = "ipc:///run/zmq/bus_in"
OUT = "ipc:///run/zmq/bus_out"


def main():
    ctx = zmq.Context.instance()
    with ctx.socket(zmq.XSUB) as xsub, ctx.socket(zmq.XPUB) as xpub:
        xsub.bind(IN)
        xpub.bind(OUT)

        print("[BROKER] up")

        zmq.proxy(xsub, xpub)


if __name__ == "__main__":
    main()
