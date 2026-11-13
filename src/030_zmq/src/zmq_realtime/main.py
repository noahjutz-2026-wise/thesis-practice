import itertools
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import zmq

XPUB_ADDR_INPROC = "inproc://xpub"
XSUB_ADDR_INPROC = "inproc://xsub"

XPUB_ADDR_IPC = "ipc://ipc_xpub"
XSUB_ADDR_IPC = "ipc://ipc_xsub"

XPUB_ADDR_TCP = "tcp://127.0.0.1:5556"
XSUB_ADDR_TCP = "tcp://127.0.0.1:5555"

XPUB_ADDR = XPUB_ADDR_TCP
XSUB_ADDR = XSUB_ADDR_TCP


def len_subscriber(sub):
    print("len sub")
    while True:
        outs = []
        t0 = time.time()
        while True:
            try:
                r = sub.recv_pyobj(zmq.EAGAIN)
                outs.append(r)
            except zmq.Again:
                break
        t1 = time.time()
        yield t1 - t0, len(outs)
        time.sleep(0.01)


def delay_subscriber(sub):
    print("time sub")
    while True:
        id, t = sub.recv_pyobj()
        t2 = time.time()
        yield id, t2 - t
        time.sleep(1)


def num_subscriber(sub):
    print("num sub")
    while True:
        id, i = sub.recv_pyobj()
        yield id, i
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


N_ITER = 5_000
TRANSPORTS = ["inproc", "ipc", "tcp"]
ADDR_MAP = {
    "inproc": ("inproc://xpub", "inproc://xsub"),
    "ipc":    ("ipc://ipc_xpub", "ipc://ipc_xsub"),
    "tcp":    ("tcp://127.0.0.1:5556", "tcp://127.0.0.1:5555"),
}
PALETTE = {"inproc": "#4CAF50", "ipc": "#2196F3", "tcp": "#FF5722"}


def benchmark(transport: str) -> tuple[list[float], list[int]]:
    print(f"[{transport}] starting…")
    xpub_addr, xsub_addr = ADDR_MAP[transport]
    ctx = zmq.Context()

    xsub = ctx.socket(zmq.XSUB)
    xpub = ctx.socket(zmq.XPUB)
    xpub.setsockopt(zmq.SNDHWM, 1)
    xpub.bind(xpub_addr)
    xsub.bind(xsub_addr)

    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVHWM, 1)
    sub.connect(xpub_addr)

    pubs = []
    for i in range(100):
        pub = ctx.socket(zmq.PUB)
        pub.connect(xsub_addr)
        pubs.append(pub)

    t_proxy = threading.Thread(target=run_proxy, args=(xsub, xpub), daemon=True)
    t_proxy.start()
    for i, pub in enumerate(pubs):
        threading.Thread(target=time_publisher, args=(pub,), kwargs={"id": i}, daemon=True).start()

    delays, drain_lens = [], []
    for d, l in itertools.islice(len_subscriber(sub), N_ITER):
        delays.append(d)
        drain_lens.append(l)
        if len(delays) % 1000 == 0:
            print(f"  [{transport}] {len(delays)}/{N_ITER}")

    for s in pubs + [sub, xpub, xsub]:
        s.close(linger=0)
    ctx.term()
    print(f"[{transport}] done")
    return delays, drain_lens


def plot(results: dict) -> None:
    transports = list(results.keys())
    colors = [PALETTE[t] for t in transports]

    delay_data = [np.array(results[t][0]) * 1e3 for t in transports]   # → ms
    drain_data = [np.array(results[t][1]) for t in transports]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"ZMQ Transport Benchmark  ({N_ITER:,} iterations / transport)\n"
                 "100 PUBs → XSUB–XPUB proxy → SUB  |  10 ms sweep", fontsize=11)

    for ax, data, ylabel, title, log in [
        (ax1, delay_data, "Drain time (ms)",       "Drain time distribution",       True),
        (ax2, drain_data, "Messages drained",       "Queue drain-length distribution", False),
    ]:
        parts = ax.violinplot(data, positions=range(len(transports)),
                              showmedians=True, showextrema=True)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)
            body.set_edgecolor("black")
            body.set_linewidth(0.8)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("black")

        rng = np.random.default_rng(42)
        for i, (d, color) in enumerate(zip(data, colors)):
            sample = rng.choice(d, size=min(len(d), 1500), replace=False)
            jitter = rng.uniform(-0.07, 0.07, size=len(sample))
            ax.scatter(i + jitter, sample, s=2, alpha=0.2, color=color, zorder=2)

        ax.set_xticks(range(len(transports)))
        ax.set_xticklabels(transports, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if log:
            ax.set_yscale("log")

    plt.tight_layout()
    out = "zmq_transport_benchmark.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.show()


def main():
    results = {}
    for transport in TRANSPORTS:
        results[transport] = benchmark(transport)
    plot(results)


if __name__ == "__main__":
    main()
