# PyZMQ Experiments

Results:

1. `zmq.CONFLATE` flag only has an effect on clients (`socket.connect()`), not on servers (`socket.bind()`). Independent of whether it is a publisher or subscriber. Applicable socket types are listed in [ZMQ_CONFLATE Docs](https://api.zeromq.org/4-2:zmq-setsockopt).
2. `zmq.CONFLATE` causes unbounded rcv and snd pipes, causing memory leak if not drained
