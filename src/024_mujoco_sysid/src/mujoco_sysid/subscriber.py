import zmq

from mujoco_sysid.broker import OUT


def main():
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(OUT)

    while True:
        x = sub.recv_pyobj()
        print(x)


if __name__ == "__main__":
    main()
