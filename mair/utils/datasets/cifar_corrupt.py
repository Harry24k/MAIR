"""
Modified from https://github.com/google-research/augmix
CREDIT: https://github.com/Harry24k
"""
import os
import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def corrupt_cifar(root, data_name, test_data, corruption):
    if data_name == "CIFAR10":
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        tar_name = "CIFAR-10-C"
    elif data_name == "CIFAR100":
        url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
        tar_name = "CIFAR-100-C"
    else:
        raise ValueError(data_name + " is not valid")

    if root[-1] == "/":
        root = root[:-1]

    if os.path.exists(os.path.join(root, tar_name + ".tar")):
        print("Files already downloaded and verified")
    else:
        download_and_extract_archive(
            download_root=root, url=url, filename=tar_name + ".tar", md5=None
        )

    base_path = root + "/" + tar_name + "/"
    test_data.data = np.load(base_path + corruption + ".npy")
    test_data.targets = torch.LongTensor(np.load(base_path + "labels.npy"))

    return test_data
