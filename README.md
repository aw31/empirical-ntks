# Empirical NTKs in PyTorch

This repository contains code for efficiently computing empirical NTKs and is published alongside our paper ["More Than a Toy: Random Matrix Models Predict How Real-World Neural Representations Generalize"](https://arxiv.org/abs/2203.06176) (ICML 2022).

## Usage

The following command computes the empirical NTK for a subset of CIFAR-10 (specifically, on the first 2,500 train samples and the first 1,000 test samples). The output is a 3,500 x 2,500 matrix.
```
python3 ntk.py CIFAR-10_0_2500_0_1000 resnet-18_pretrained --workers-per-device 2 --grad-chunksize 1900000 --mm-col-chunksize 20000 --loader-batch-size 50 --loader-num-workers 12
```

The following command computes the empirical NTK for all of CIFAR-10. The output is a 60,000 x 50,000 matrix.
```
python3 ~/empirical-ntks/ntk.py CIFAR-10 resnet-18_pretrained --workers-per-device 4 --grad-chunksize 1900000 --mm-col-chunksize 20000 --loader-batch-size 50 --loader-num-workers 12
```

To work with other datasets or models, see `utils.py` for further options.


## Implementation

We pursue a very simple strategy for computing the empirical NTK: compute the `N x P` Jacobian matrix (for `N` samples and `P` parameters) and multiply it with its transpose. To make this computation feasible, we compute the Jacobian matrix in chunks along the `P` axis with matrices of size `N x P0` (where `P0` is set by `--grad-chunksize`). We store this (still large) matrix in RAM. For each chunk, we then compute the `N x N` matrix obtained by multiplying each chunk by its transpose; for each such computation, we again chunk along the `P` axis (and optionally along the `N` axis), sending each matrix multiplication to the GPU. This latter matrix multiplication step is typically the bottleneck in computation time.

By optimizing data transfer, increasing GPU utilization, and parallelizing with care, our implementation improves significantly over naive baselines. See `ntk.py` for implementation details.

## Performance

Our library computes an empirical NTK (60,000 x 50,000) for a ResNet-18 over CIFAR-10 at `float32` precision in 43 minutes (<1e-6 seconds per NTK entry) on a machine with four A100 GPUs and 755GB RAM.

## Citation

If you find this code useful in your research, please consider citing our paper:
```
@inproceedings{wei2022more,
  title = {More Than a Toy: Random Matrix Models Predict How Real-World Neural Representations Generalize},
  author = {Wei, Alexander and Hu, Wei and Steinhardt, Jacob},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  year = {2022}
}
```
