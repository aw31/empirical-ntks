import copy
import logging
import os
import pathlib
import threading
import time
import torch

from torch.multiprocessing import Process, Queue
from tqdm.auto import tqdm

from multiqueue_worker import multiqueue_worker
from utils import init_torch, humanize_units

local = threading.local()


def wrap_loader(loader):
    batch_start = 0
    for batch in loader:
        batch_len = batch[0].size()[0]
        batch_stop = batch_start + batch_len
        yield (batch, (slice(batch_start, batch_stop), batch_len))
        batch_start = batch_stop


def _init_compute_gradients(model, params_slice, buffer_size):
    if not "model" in local.__dict__:
        local.model = model.cuda()

    if not "grad" in local.__dict__:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_size = param_count + (-param_count % buffer_size[1])
        local.grad = torch.zeros(grad_size, dtype=torch.float, device="cuda")

    slot_start = 0
    local.active_params = []
    for param, local_param in zip(model.parameters(), local.model.parameters()):
        if not param.requires_grad:
            continue
        local_param.requires_grad = False
        slot_stop = slot_start + param.numel()
        if not (slot_stop <= params_slice.start or slot_start >= params_slice.stop):
            local.active_params.append((slice(slot_start, slot_stop), local_param))
            local_param.requires_grad = True
        slot_start = slot_stop

    local.params_slice = params_slice


def _compute_gradients(model, params_slice, buffer_size, data, batch_info):
    if not "params_slice" in local.__dict__ or local.params_slice != params_slice:
        _init_compute_gradients(model, params_slice, buffer_size)

    local.grad.zero_()
    gpu_buffer = torch.zeros(buffer_size, device="cuda")

    data = data.cuda()
    for i in range(0, data.size()[0]):
        local.model.zero_grad(set_to_none=True)
        local.model.forward(data[i : i + 1])[0].backward()
        for slot, param in local.active_params:
            local.grad[slot] = param.grad.flatten()
        gpu_buffer[i] = local.grad[local.params_slice]

    del data
    torch.cuda.empty_cache()

    return gpu_buffer, batch_info


def compute_gradients(in_queue, out_queue, model, params_slice, loader, out):
    buffer_size = (loader.batch_size, out.size()[1])

    in_flight = 0
    pbar = tqdm(total=len(loader.dataset))

    for batch, batch_info in wrap_loader(loader):
        data, _ = batch
        args = (model, params_slice, buffer_size, data.clone(), batch_info)
        in_queue.put((_compute_gradients, args))
        in_flight += 1

        if in_flight >= 36:
            gpu_buffer, batch_info = out_queue.get()
            gpu_buffer_clone = gpu_buffer.clone()
            batch_slice, batch_len = batch_info
            del gpu_buffer

            out[batch_slice].copy_(gpu_buffer_clone[:batch_len])
            del gpu_buffer_clone
            in_flight -= 1
            pbar.update(batch_len)

    while in_flight > 0:
        gpu_buffer, batch_info = out_queue.get()
        gpu_buffer_clone = gpu_buffer.clone()
        batch_slice, batch_len = batch_info
        del gpu_buffer

        out[batch_slice].copy_(gpu_buffer_clone[:batch_len])
        del gpu_buffer_clone
        in_flight -= 1
        pbar.update(batch_len)

    pbar.close()


def _compute_XXt(chunk, buffer_size, buffer_dtype, train_slice, test_slice):
    if not "buffer" in local.__dict__:
        local.buffer = torch.zeros(buffer_size, dtype=buffer_dtype, device="cuda")

    chunk = chunk.to(local.buffer)
    local.buffer.addmm_(chunk[test_slice], chunk[train_slice].T)


def _return_XXt_buffer():
    assert "buffer" in local.__dict__
    return local.buffer


def _clear_XXt_buffer():
    assert "buffer" in local.__dict__
    del local.buffer
    torch.cuda.empty_cache()


def compute_XXt(
    in_queue_XXt, in_queues_devices, out_queue, X, out, row_chunksize, col_chunksize
):
    in_flight = 0
    train_slice = slice(0, out.size()[1])
    for i in range(0, X.size()[0], row_chunksize):
        test_slice = slice(i, i + row_chunksize)
        for j in tqdm(range(0, X.size()[1], col_chunksize)):
            chunk = X[:, j : j + col_chunksize].clone()
            args = (
                chunk,
                out[test_slice].shape,
                out.dtype,
                train_slice,
                test_slice,
            )
            in_queue_XXt.put((_compute_XXt, args))
            in_flight += 1

            if in_flight >= 3 * len(in_queues_devices):
                _ = out_queue.get()
                in_flight -= 1

        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_return_XXt_buffer, ()))
            in_flight += 1

        while in_flight > 0:
            gpu_buffer = out_queue.get()
            out[test_slice].add_(gpu_buffer.cpu())
            gpu_buffer.zero_()
            del gpu_buffer
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_clear_XXt_buffer, ()))
            in_flight += 1

        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1


def compute_ntk(
    model,
    train_set,
    test_set,
    num_devices=None,
    workers_per_device=1,
    grad_chunksize=None,
    mm_col_chunksize=None,
    mm_row_chunksize=None,
    loader_kwargs={},
    pin_memory=True,
    ntk_dtype=torch.double,
    init_torch_kwargs={},
):
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    if grad_chunksize is None:
        assert False  # TODO: Tune automatically?
    if mm_col_chunksize is None:
        assert False  # TODO: Tune automatically?
    if mm_row_chunksize is None:
        mm_row_chunksize = 1000000000  # Don't chunk rows by default

    if not "persistent_workers" in loader_kwargs:
        loader_kwargs["persistent_workers"] = True

    logging.info(f"Executing on {num_devices} device(s)")

    num_workers = num_devices * workers_per_device
    in_queue_grad = Queue()
    in_queue_XXt = Queue()
    in_queues_devices = [Queue() for _ in range(num_devices)]
    out_queue = Queue()
    for i in range(num_workers):
        device = i % num_devices
        i_in_queues = [in_queue_grad]
        if i < num_devices:
            i_in_queues.append(in_queue_XXt)
            i_in_queues.append(in_queues_devices[i])
        args = (device, init_torch_kwargs, i_in_queues, out_queue)
        Process(target=multiqueue_worker, args=args).start()

    model.zero_grad(set_to_none=True)
    model.eval()

    train_test_sets = torch.utils.data.ConcatDataset([train_set, test_set])
    loader = torch.utils.data.DataLoader(train_test_sets, **loader_kwargs)

    grads_bytes = 4 * len(loader.dataset) * grad_chunksize
    grad_buffer_bytes = 4 * loader.batch_size * grad_chunksize
    mm_buffer_bytes = 4 * len(loader.dataset) * mm_col_chunksize
    logging.info(f"Pinning gradient Tensor of size {humanize_units(grads_bytes)}")
    logging.info(f"Using gradient buffers of size {humanize_units(grad_buffer_bytes)}")
    logging.info(f"Using matmul buffers of size {humanize_units(mm_buffer_bytes)}")

    pin_begin = time.time()
    grads_size = (len(loader.dataset), grad_chunksize)
    grads = torch.zeros(grads_size, dtype=torch.float, pin_memory=pin_memory)
    pin_end = time.time()
    logging.info(f"Allocated grads in {int(pin_end - pin_begin)}s")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_batches = (param_count - 1) // grad_chunksize + 1

    ntk = torch.zeros((len(train_test_sets), len(train_set)), dtype=ntk_dtype)

    for i, params_start in enumerate(range(0, param_count, grad_chunksize)):
        logging.info(f"Starting batch {i + 1}/{param_batches}")

        params_stop = params_start + grad_chunksize
        params_slice = slice(params_start, params_stop)

        grads_begin = time.time()
        compute_gradients(in_queue_grad, out_queue, model, params_slice, loader, grads)
        grads_end = time.time()
        logging.info(f"Computed grads in {int(grads_end - grads_begin)}s")
        torch.cuda.empty_cache()

        ntk_begin = time.time()
        compute_XXt(
            in_queue_XXt,
            in_queues_devices,
            out_queue,
            grads,
            ntk,
            mm_row_chunksize,
            mm_col_chunksize,
        )
        ntk_end = time.time()
        logging.info(f"Computed NTK in {int(ntk_end - ntk_begin)}s")
        torch.cuda.empty_cache()

    for i in range(num_workers):
        in_queue_grad.put(None)

    return ntk


def save_ntk(ntk, savedir, handle):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    path = savedir / f"{handle}_ntk-v2_{timestamp}.pt"
    torch.save(ntk, path)
    logging.info(f"Saved NTK to {path}")


def load_ntk(savedir, handle, map_location=None):
    savedir = pathlib.Path(savedir).resolve()
    files = list(savedir.glob(f"{handle}_ntk-v2_*.pt"))

    assert len(files) > 0, f"No matching files for {handle}_ntk-v2_*.pt in {savedir}!"
    if len(files) > 1:
        logging.warning(f"Multiple matching NTKs found!")

    files = sorted(files)
    logging.info(f"Loading NTK from {files[-1]}")
    ntk = torch.load(files[-1], map_location=map_location)
    return ntk


if __name__ == "__main__":
    import argparse
    import pprint
    from torch.multiprocessing import set_start_method, set_sharing_strategy
    from utils import init_logging, load_model, load_dataset

    # Set up
    set_start_method("spawn")
    set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--datadir", type=str, default="./datasets")
    parser.add_argument("--savedir", type=str, default="./ntks")
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--grad-chunksize", type=int)
    parser.add_argument("--mm-col-chunksize", type=int)
    parser.add_argument("--ntk-dtype", type=str, default="float32")
    parser.add_argument("--loader-batch-size", type=int)
    parser.add_argument("--loader-num-workers", type=int)
    parser.add_argument("--no-pinned-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false"
    )
    args = parser.parse_args()

    if args.logdir:
        init_logging("ntk", args.logdir)
    logging.info(f"args =\n{pprint.pformat(vars(args))}")

    # Initialize torch
    init_torch_kwargs = {
        "allow_tf32": args.allow_tf32,
        "benchmark": args.benchmark,
        "deterministic": args.deterministic,
    }
    init_torch(**init_torch_kwargs, verbose=True)

    # Initialize model
    model = load_model(args.model)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_batches = (param_count - 1) // args.grad_chunksize + 1
    logging.info(f"Splitting {param_count} parameters into {param_batches} batches")

    # Initialize datasets
    datadir = pathlib.Path(args.datadir)
    train_set = load_dataset(datadir, args.dataset, "train")
    test_set = load_dataset(datadir, args.dataset, "test")

    # Compute NTK
    loader_kwargs = {
        "batch_size": args.loader_batch_size,
        "num_workers": args.loader_num_workers,
        "persistent_workers": False if args.loader_num_workers == 0 else None,
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    kwargs = {
        "workers_per_device": args.workers_per_device,
        "grad_chunksize": args.grad_chunksize,
        "mm_col_chunksize": args.mm_col_chunksize,
        "loader_kwargs": loader_kwargs,
        "pin_memory": args.pin_memory,
        "init_torch_kwargs": init_torch_kwargs,
        "ntk_dtype": torch.float32 if args.ntk_dtype == "float32" else torch.float64,
    }
    ntk = compute_ntk(model, train_set, test_set, **kwargs)

    # Save NTK
    save_ntk(ntk, args.savedir, f"{args.dataset}_{args.model}")

    logging.info(f"{ntk.size() = }")
    logging.info(f"ntk =\n{ntk}")
