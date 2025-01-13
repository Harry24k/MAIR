"""
Modified from
https://github.com/Clockware/nn-tiny-imagenet-200/blob/master/nn-tiny-imagenet-200.py
CREDIT: https://github.com/Clockware
"""
import zipfile
import os
from urllib.request import urlretrieve
from shutil import copyfile

import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_and_extract_archive


class TinyImageNet:
    def __init__(self, root="./data", train=True, transform=None):

        if root[-1] == "/":
            root = root[:-1]

        self._ensure_dataset_loaded(root)

        if train:
            self.data = dsets.ImageFolder(
                root + "/tiny-imagenet-200/train", transform=transform
            )
        else:
            self.data = dsets.ImageFolder(
                root + "/tiny-imagenet-200/val_fixed", transform=transform
            )

    def _download_dataset(
        self,
        path,
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tar_name="tiny-imagenet-200.zip",
    ):
        download_and_extract_archive(
            download_root=path, url=url, filename=tar_name, md5=None
        )

    def _ensure_dataset_loaded(self, root):
        val_fixed_folder = root + "/tiny-imagenet-200/val_fixed"
        if os.path.exists(val_fixed_folder):
            return

        self._download_dataset(root)
        os.mkdir(val_fixed_folder)

        with open(root + "/tiny-imagenet-200/val/val_annotations.txt") as f:
            for line in f.readlines():
                fields = line.split()

                file_name = fields[0]
                clazz = fields[1]

                class_folder = root + "/tiny-imagenet-200/val_fixed/" + clazz
                if not os.path.exists(class_folder):
                    os.mkdir(class_folder)

                original_image_path = (
                    root + "/tiny-imagenet-200/val/images/" + file_name
                )
                copied_image_path = class_folder + "/" + file_name

                copyfile(original_image_path, copied_image_path)
