# PyZMQ Experiments

## Results

[`zmq.CONFLATE`](https://api.zeromq.org/4-2:zmq-setsockopt)

1. `zmq.CONFLATE` flag only has an effect on clients (`socket.connect()`), not on servers (`socket.bind()`). Independent of whether it is a publisher or subscriber. Applicable socket types are listed in docs.
2. `zmq.CONFLATE` causes unbounded rcv and snd pipes, causing memory leak if not drained

`zmq.RCVHWM` / `zmq.SNDHWM`

1. In proxy setup, 2 reads drain pipeline. likely: outbound xpub buffer and inbound sub buffer.
2. send to full buffer is discarded. solution: drain.
3. 100 concurrent publishers can overload a subscriber, making draining impossible. solution: timestamped drain
