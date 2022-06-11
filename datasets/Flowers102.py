import json
from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

from torchvision.datasets.utils import (
    verify_str_arg,
    download_and_extract_archive,
    download_url,
)
from torchvision.datasets import VisionDataset


class Flowers102(VisionDataset):
    """The Flowers-102 Data Set https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

    This set contains images of flowers belonging to 102 different categories.
    The images were acquired by searching the web and taking pictures. There are a
    minimum of 40 images for each category.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    _DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    _LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    _SPLITS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    _SPLITS_MAP = {
        "train": "trn",
        "val": "val",
        "test": "tst",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root)
        self._image_labels = self._base_folder / "imagelabels.mat"
        self._set_id = self._base_folder / "setid.mat"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        from scipy import io

        image_labels = io.loadmat(self._image_labels)["labels"][0]
        set_id = io.loadmat(self._set_id)

        split_ids = set_id[f"{self._SPLITS_MAP[self._split]}id"][0]
        self._labels = [image_labels[i - 1] - 1 for i in split_ids]
        self._image_files = [
            self._images_folder / f"image_{i:05d}.jpg" for i in split_ids
        ]

        if self._split == "train":
            val_split_ids = set_id[f"valid"][0]
            val_labels = [image_labels[i - 1] - 1 for i in val_split_ids]
            val_image_files = [
                self._images_folder / f"image_{i:05d}.jpg" for i in val_split_ids
            ]
            self._labels += val_labels
            self._image_files += val_image_files

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return (
            self._image_labels.exists()
            and self._image_labels.is_file()
            and self._set_id.exists()
            and self._set_id.is_file()
            and self._images_folder.exists()
            and self._images_folder.is_dir()
        )

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATA_URL, download_root=self.root)
        download_url(self._LABELS_URL, root=self.root)
        download_url(self._SPLITS_URL, root=self.root)
