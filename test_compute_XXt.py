import os
import threading
import time
import torch

from tqdm.auto import tqdm

from torch.multiprocessing import Process, Queue

from multiqueue_worker import multiqueue_worker

local = threading.local()

from ntk import compute_XXt


def test_compute_XXt():
    # Initialize torch
    init_torch_kwargs = {
        "allow_tf32": False,
        "benchmark": False,
        "deterministic": True,
    }

    num_devices = 4
    in_queue_XXt = Queue()
    in_queues_devices = [Queue() for _ in range(num_devices)]
    out_queue = Queue()
    for i in range(num_devices):
        device = i % num_devices
        args = (
            device,
            init_torch_kwargs,
            [in_queue_XXt, in_queues_devices[i]],
            out_queue,
        )
        Process(target=multiqueue_worker, args=args).start()

    X = torch.randn(100, 1000, dtype=torch.float64)
    out = torch.zeros(100, 70, dtype=torch.float64)
    row_chunksize = 40
    col_chunksize = 10

    compute_XXt(
        in_queue_XXt, in_queues_devices, out_queue, X, out, row_chunksize, col_chunksize
    )

    print(out)
    print((X @ X.T)[:, :70])
    assert (out - (X @ X.T)[:, :70]).abs().max() < 1e-8

    for i in range(num_devices):
        in_queues_devices[i].put(None)


if __name__ == "__main__":
    test_compute_XXt()
