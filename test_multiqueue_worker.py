import os
import time

from torch.multiprocessing import Process, Queue

from multiqueue_worker import multiqueue_worker


def foo():
    time.sleep(5)
    return f"foo ({os.getpid()})"


def bar():
    time.sleep(1)
    return f"bar ({os.getpid()})"


def test_multiqueue_worker():
    # Initialize torch
    init_torch_kwargs = {
        "allow_tf32": False,
        "benchmark": False,
        "deterministic": True,
    }

    num_workers = 8
    num_devices = 2
    in_queues = [Queue(), Queue()]
    out_queue = Queue()
    for i in range(num_workers):
        device = i % num_devices
        args = (device, init_torch_kwargs, in_queues, out_queue)
        Process(target=multiqueue_worker, args=args).start()

    for _ in range(4):
        in_queues[0].put((foo, ()))
    for _ in range(12):
        in_queues[1].put((bar, ()))
    for _ in range(16):
        print(out_queue.get())

    for i in range(num_workers):
        in_queues[0].put(None)


if __name__ == "__main__":
    test_multiqueue_worker()
