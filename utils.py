import logging
import pathlib
import sys
import time

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from FGVCAircraft import FGVCAircraft
from Food101 import Food101
from Flowers102 import Flowers102

# def print_gpu_stats():
#     t = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
#     r = torch.cuda.memory_reserved(0) / 1024 ** 3
#     a = torch.cuda.memory_allocated(0) / 1024 ** 3
#     print(f'{t:.2f} GiB total capacity; {a:.2f} GiB already allocated; {r:.2f} GiB reserved in total by PyTorch')

# From https://stackoverflow.com/a/1094933
def humanize_units(size, unit="B"):
    for prefix in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if size < 1024.0 or prefix == "Pi":
            break
        size /= 1024.0
    return f"{size:.1f}{prefix}"


def init_torch(allow_tf32=False, benchmark=False, deterministic=True, verbose=False):
    # Disable tf32 in favor of more accurate gradients
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # Benchmarking can lead to non-determinism
    torch.backends.cudnn.benchmark = benchmark

    # Ensure repeated gradient calculations are consistent
    torch.backends.cudnn.deterministic = deterministic

    if verbose:
        logging.info(f"{torch.backends.cuda.matmul.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.benchmark = }")
        logging.info(f"{torch.backends.cudnn.deterministic = }")


def init_logging(handle, logdir):
    logdir = pathlib.Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    filename = logdir / f"{handle}-{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"Logging to {filename}")


def load_model(name):
    rng_state = torch.get_rng_state()
    if name == "resnet-18_init":
        torch.manual_seed(438)
        model = models.resnet18()
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-18_pretrained":
        torch.manual_seed(438)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-34_init":
        torch.manual_seed(438)
        model = models.resnet34()
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-34_pretrained":
        torch.manual_seed(438)
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 1)
    elif name == "resnet-50_init":
        torch.manual_seed(438)
        model = models.resnet50()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-50_pretrained":
        torch.manual_seed(438)
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-101_init":
        torch.manual_seed(438)
        model = models.resnet101()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnet-101_pretrained":
        torch.manual_seed(438)
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "resnext-101-32x8d_init":
        torch.manual_seed(438)
        model = models.resnet101_32x8d()
        model.fc = nn.Linear(2048, 1)
    elif name == "resnext-101-32x8d_pretrained":
        torch.manual_seed(438)
        model = models.resnet101_32x8d(pretrained=True)
        model.fc = nn.Linear(2048, 1)
    elif name == "efficientnet-b7_init":
        torch.manual_seed(438)
        model = models.efficientnet_b7()
        model.classifier[1] = nn.Linear(2560, 1)
    elif name == "efficientnet-b7_pretrained":
        torch.manual_seed(438)
        model = models.efficientnet_b7(pretrained=True)
        model.classifier[1] = nn.Linear(2560, 1)
    else:
        assert False
    torch.set_rng_state(rng_state)

    return model


def load_FakeData():
    transform = transforms.ToTensor()
    dataset = datasets.FakeData(size=250, transform=transform)
    return dataset


def load_CIFAR10(datadir, split):
    root = str(datadir / "CIFAR-10")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)
    return dataset


def load_CIFAR100(datadir, split):
    root = str(datadir / "CIFAR-100")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.CIFAR100(root, train=train, transform=transform, download=True)
    return dataset


def load_SVHN(datadir, split):
    root = str(datadir / "SVHN")
    mean = [0.4380, 0.4440, 0.4730]
    std = [0.1751, 0.1771, 0.1744]
    transform = [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.SVHN(root, split=split, transform=transform, download=True)
    return dataset


def load_FashionMNIST(datadir, split):
    root = str(datadir / "FashionMNIST")
    train = split == "train"
    mean = [0.2860]
    std = [0.3530]
    transform = [
        transforms.Grayscale(3),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = datasets.FashionMNIST(
        root, train=train, transform=transform, download=True
    )
    return dataset


def load_FGVCAircraft(datadir, split):
    root = str(datadir / "FGVCAircraft")
    train = split == "train"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = FGVCAircraft(root, train=train, transform=transform, download=True)
    return dataset


def load_Food101(datadir, split):
    root = str(datadir / "Food-101")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = Food101(root, split=split, transform=transform, download=True)
    return dataset


def load_Flowers102(datadir, split):
    root = str(datadir / "Flowers-102")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = [
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    transform = transforms.Compose(transform)
    dataset = Flowers102(root, split=split, transform=transform, download=True)
    return dataset


def load_subset(name, split, dataset):
    train = split == "train"
    _, train_begin, train_end, test_begin, test_end = name.split("_")
    subset = (train_begin, train_end) if train else (test_begin, test_end)
    subset_begin, subset_end = map(int, subset)

    assert subset_begin >= 0, f"{subset_begin} < 0"
    assert subset_begin < subset_end, f"{subset_begin} >= {subset_end}"
    assert subset_end <= len(dataset), f"{subset_end} > {len(dataset)}"

    dataset = torch.utils.data.Subset(dataset, range(subset_begin, subset_end))
    return dataset


def load_dataset(datadir, name, split):
    if name == "FakeData":
        dataset = load_FakeData()
    elif name == "CIFAR-10":
        dataset = load_CIFAR10(datadir, split)
    elif name.startswith("CIFAR-10_"):
        dataset = load_CIFAR10(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "CIFAR-100":
        dataset = load_CIFAR100(datadir, split)
    elif name.startswith("CIFAR-100_"):
        dataset = load_CIFAR100(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "SVHN":
        dataset = load_SVHN(datadir, split)
    elif name.startswith("SVHN_"):
        dataset = load_SVHN(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "FashionMNIST":
        dataset = load_FashionMNIST(datadir, split)
    elif name.startswith("FashionMNIST_"):
        dataset = load_FashionMNIST(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "FGVCAircraft":
        dataset = load_FGVCAircraft(datadir, split)
    elif name.startswith("FGVCAircraft_"):
        dataset = load_FGVCAircraft(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "Food-101":
        dataset = load_Food101(datadir, split)
    elif name.startswith("Food-101_"):
        dataset = load_Food101(datadir, split)
        dataset = load_subset(name, split, dataset)
    elif name == "Flowers-102":
        dataset = load_Flowers102(datadir, split)
    elif name.startswith("Flowers-102_"):
        dataset = load_Flowers102(datadir, split)
        dataset = load_subset(name, split, dataset)
    else:
        assert False

    return dataset


def num_classes_of(name):
    if name == "FakeData":
        num_classes = 0
    elif name == "CIFAR-10" or name.startswith("CIFAR-10_"):
        num_classes = 10
    elif name == "CIFAR-100" or name.startswith("CIFAR-100_"):
        num_classes = 100
    elif name == "SVHN" or name.startswith("SVHN_"):
        num_classes = 10
    elif name == "FashionMNIST" or name.startswith("FashionMNIST_"):
        num_classes = 10
    elif name == "FGVCAircraft" or name.startswith("FGVCAircraft_"):
        num_classes = 102
    elif name == "Food-101" or name.startswith("Food-101_"):
        num_classes = 101
    elif name == "Flowers-102" or name.startswith("Flowers-102_"):
        num_classes = 102
    else:
        assert False

    return num_classes
