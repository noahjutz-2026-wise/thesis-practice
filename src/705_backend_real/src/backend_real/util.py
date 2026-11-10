import zmq


def drain(sub):
    while True:
        try:
            _ = sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
