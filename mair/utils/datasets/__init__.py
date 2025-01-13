import os.path
import numpy as np
from copy import deepcopy
import logging

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from .tinyimagenet import TinyImageNet
from .mnistm import MNISTM
from .cifar_unsup import SemiSupervisedSampler, CIFARunsup
from .cifar_corrupt import CORRUPTIONS, corrupt_cifar
from .cifar10h import probs
from .imagenet_natural_adv import get_imagnet_natural_adv
from .imagenet_renditions import get_imagnet_renditions


class Datasets:
    def __init__(
        self,
        data_name,
        root="./data",
        val_info=None,
        val_seed=0,
        label_filter=None,
        train_transform=None,
        test_transform=None,
        val_transform=None,
        corruption=None,
        *args,
        **kwargs,
    ):

        self.val_info = val_info
        self.val_seed = val_seed

        self.train_data_sup = None
        self.train_data_unsup = None
        self.train_data = None
        self.test_data = None

        # TODO : Validation + Label filtering
        if val_info is not None:
            if label_filter is not None:
                raise ValueError("Validation + Label filtering is not supported yet.")

        # Base transform
        if (data_name == "CIFAR10") or (data_name == "CIFAR100"):
            if train_transform is None:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            if test_transform is None:
                test_transform = transforms.ToTensor()
            if val_transform is None:
                val_transform = transforms.ToTensor()

        elif data_name == "TinyImageNet":
            if train_transform is None:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(64, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            if test_transform is None:
                test_transform = transforms.ToTensor()
            if val_transform is None:
                val_transform = transforms.ToTensor()

        elif "ImageNet" in data_name:
            if train_transform is None:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            if test_transform is None:
                test_transform = transforms.Compose(
                    [transforms.Resize((224, 224)), transforms.ToTensor(),]
                )
            if val_transform is None:
                val_transform = transforms.Compose(
                    [transforms.Resize((224, 224)), transforms.ToTensor(),]
                )

        else:
            if train_transform is None:
                logging.warning("transforms.ToTensor() is used as a train transform.")
                train_transform = transforms.Compose([transforms.ToTensor(),])
            if test_transform is None:
                logging.warning("transforms.ToTensor() is used as a test transform.")
                test_transform = transforms.Compose([transforms.ToTensor(),])
            if val_transform is None:
                logging.warning("transforms.ToTensor() is used as a val transform.")
                val_transform = transforms.Compose([transforms.ToTensor(),])

        # Set data by data_name
        set_data_method = getattr(self, "_set_data_" + data_name, None)
        if set_data_method is not None:
            set_data_method(root, train_transform, test_transform, *args, **kwargs)
        else:
            raise ValueError(data_name + " is not valid")

        # Corruption for only CIFAR:
        if corruption is not None:
            assert "CIFAR" in data_name
            assert corruption in CORRUPTIONS
            print("Corruption is only applied to the test dataset.")
            self.train_data = EmptyDataset()
            self.test_data = corrupt_cifar(root, data_name, self.test_data, corruption)

        self.data_name = data_name

        if (self.val_info is not None) and (len(self.train_data) > 0):
            # For unsup datasets...
            if self.train_data_sup is not None:
                self.train_data = self.train_data_sup

            max_len = len(self.train_data)
            if isinstance(self.val_info, float):
                if self.val_info <= 0 or self.val_info >= 1:
                    raise ValueError(
                        "The ratio of validation set must be in the range of (0, 1)."
                    )
                else:
                    self.val_len = int(max_len * self.val_info)
                    self.val_idx = (
                        np.random.RandomState(seed=self.val_seed)
                        .permutation(max_len)[: self.val_len]
                        .tolist()
                    )
            elif isinstance(self.val_info, int):
                if self.val_info <= 0 or self.val_info >= max_len:
                    raise ValueError(
                        "The number of validation set must be in the range of (0, len(train_data))."
                    )
                else:
                    self.val_len = self.val_info
                    self.val_idx = (
                        np.random.RandomState(seed=self.val_seed)
                        .permutation(max_len)[: self.val_len]
                        .tolist()
                    )
            elif isinstance(self.val_info, list):
                self.val_len = len(self.val_info)
                self.val_idx = self.val_info
                pass
            else:
                raise ValueError("val_info must be the one of [int, float or list].")

            copy_train_data = deepcopy(self.train_data)
            self.val_data = Subset(copy_train_data, self.val_idx)
            self.val_data.dataset.transform = val_transform

            self.train_idx = list(set(range(len(self.train_data))) - set(self.val_idx))
            self.train_data = Subset(self.train_data, self.train_idx)
            # For unsup datasets...
            if self.train_data_sup is not None:
                self.train_data = ConcatDataset(
                    [self.train_data, self.train_data_unsup]
                )

            self.train_len = len(self.train_data)
            self.test_len = len(self.test_data)

            print("Data Loaded (w/ Validation Set)!")
            print("Train Data Length :", self.train_len)
            print("Val Data Length :", self.val_len)
            print("Test Data Length :", self.test_len)

        elif label_filter is not None:  # noqa: W191
            if data_name in ["ImageNet-O", "ImageNet-A", "ImageNet-R"]:
                raise ValueError("Label filter is not supported for %s" % data_name)

            # Tensor label to list
            self.train_len = len(self.train_data)
            self.test_len = len(self.test_data)
            if self.train_len > 0:
                if type(self.train_data.targets) is torch.Tensor:
                    self.train_data.targets = self.train_data.targets.numpy()

                filtered = []
                for (i, label) in enumerate(self.train_data.targets):
                    if label in label_filter.keys():
                        filtered.append(i)
                        self.train_data.targets[i] = label_filter[label]

                self.train_data = Subset(self.train_data, filtered)
                self.train_len = len(self.train_data)

                if type(self.test_data.targets) is torch.Tensor:
                    self.test_data.targets = self.test_data.targets.numpy()

            if self.test_len > 0:
                filtered = []
                for (i, label) in enumerate(self.test_data.targets):
                    if label in label_filter.keys():
                        filtered.append(i)
                        self.test_data.targets[i] = label_filter[label]

                self.test_data = Subset(self.test_data, filtered)
                self.test_len = len(self.test_data)

            print("Data Loaded! (w/ Label Filtering)")
            print("Train Data Length :", self.train_len)
            print("Test Data Length :", self.test_len)

        else:
            self.train_len = len(self.train_data)
            self.test_len = len(self.test_data)

            print("Data Loaded!")
            print("Train Data Length :", self.train_len)
            print("Test Data Length :", self.test_len)

    def get_len(self):
        if self.val_info is None:
            return self.train_len, self.test_len

        else:
            return self.train_len, self.val_len, self.test_len

    def get_data(self):
        if self.val_info is None:
            return self.train_data, self.test_data

        else:
            return self.train_data, self.val_data, self.test_data

    def get_balanced_sampler(self, data):
        count = {}
        for _, label in data:
            if count.get(int(label)) is None:
                count[int(label)] = 0
            count[int(label)] += 1
        nclasses = len(count.keys())
        weight_per_class = [0.0] * nclasses
        N = float(sum(list(count.values())))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(data)
        for idx, val in enumerate(data):
            weight[idx] = weight_per_class[int(val[1])]
        weight = torch.DoubleTensor(weight)
        sampler = WeightedRandomSampler(weight, len(weight))
        return sampler

    def get_loader(
        self,
        batch_size,
        drop_last_train=True,
        num_workers=0,
        shuffle_train=True,
        shuffle_val=False,
        make_balanced_train=False,
    ):
        if self.train_len > 0:
            sampler = None
            if make_balanced_train:
                sampler = self.get_balanced_sampler(self.train_data)
                shuffle_train = None
                logging.warning(
                    "shuffle_train is mutually exclusive with make_balanced_train."
                )
            self.train_loader = DataLoader(
                dataset=self.train_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_train,
                sampler=sampler,
                drop_last=drop_last_train,
            )

            # For unsup datasets...
            if self.train_data_sup is not None:
                train_batch_sampler = SemiSupervisedSampler(
                    list(range(len(self.train_data_sup))),
                    self.train_data_unsup.unsup_indices,
                    batch_size,
                    unsup_fraction=0.5,
                    num_batches=int(np.ceil(50000 / batch_size)),
                )

                self.train_loader = DataLoader(
                    dataset=self.train_data,
                    num_workers=num_workers,
                    batch_sampler=train_batch_sampler,
                )
        else:
            self.train_loader = None

        if self.test_len > 0:
            self.test_loader = DataLoader(
                dataset=self.test_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
        else:
            self.test_loader = None

        if self.val_info is not None:
            self.val_loader = DataLoader(
                dataset=self.val_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_val,
            )

            return self.train_loader, self.val_loader, self.test_loader

        return self.train_loader, self.test_loader

    def _set_data_CIFAR10(self, root, train_transform, test_transform):
        self.train_data = dsets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_CIFAR100(self, root, train_transform, test_transform):
        self.train_data = dsets.CIFAR100(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.CIFAR100(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_STL10(self, root, train_transform, test_transform):
        self.train_data = dsets.STL10(
            root=root, split="train", download=True, transform=train_transform
        )

        self.test_data = dsets.STL10(
            root=root, split="test", download=True, transform=test_transform
        )

    def _set_data_MNIST(self, root, train_transform, test_transform):
        self.train_data = dsets.MNIST(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.MNIST(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_FashionMNIST(self, root, train_transform, test_transform):
        self.train_data = dsets.FashionMNIST(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.FashionMNIST(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_SVHN(self, root, train_transform, test_transform):
        self.train_data = dsets.SVHN(
            root=root, split="train", download=True, transform=train_transform
        )

        self.test_data = dsets.SVHN(
            root=root, split="test", download=True, transform=test_transform
        )

    def _set_data_MNISTM(self, root, train_transform, test_transform):
        self.train_data = MNISTM(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = MNISTM(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_ImageNet(self, root, train_transform, test_transform):
        file_meta = "ILSVRC2012_devkit_t12.tar.gz"
        file_train = "ILSVRC2012_img_train.tar"
        file_val = "ILSVRC2012_img_val.tar"
        if root[-1] == "/":
            root = root[:-1]

        if os.path.isfile(root + "/" + file_meta):
            pass
        else:  # noqa: W191
            raise ValueError(
                "Please download ImageNet Meta file via https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz."
            )

        if os.path.isfile(root + "/" + file_train) and os.path.isfile(
            root + "/" + file_val
        ):
            pass
        elif os.path.isdir(root + "/train") and os.path.isdir(root + "/val"):
            pass
        else:
            raise ValueError(
                "Please download ImageNet files via https://academictorrents.com/collection/imagenet-2012."
            )

        self.train_data = dsets.ImageNet(
            root=root, split="train", transform=train_transform
        )

        self.test_data = dsets.ImageNet(
            root=root, split="val", transform=test_transform
        )

    def _set_data_USPS(self, root, train_transform, test_transform):
        self.train_data = dsets.USPS(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.USPS(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_TinyImageNet(self, root, train_transform, test_transform):
        self.train_data = TinyImageNet(
            root=root, train=True, transform=train_transform
        ).data

        self.test_data = TinyImageNet(
            root=root, train=False, transform=test_transform
        ).data

    def _set_data_CIFAR10U(self, root, train_transform, test_transform):
        self.train_data_sup = dsets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )

        self.train_data_unsup = CIFARunsup(
            root=root, download=True, transform=train_transform
        )

        self.train_data = ConcatDataset([self.train_data_sup, self.train_data_unsup])

        self.test_data = dsets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_CIFAR100U(self, root, train_transform, test_transform):
        self.train_data_sup = dsets.CIFAR100(
            root=root, train=True, download=True, transform=train_transform
        )

        self.train_data_unsup = CIFARunsup(
            root=root, download=True, transform=train_transform
        )

        self.train_data = ConcatDataset([self.train_data_sup, self.train_data_unsup])

        self.test_data = dsets.CIFAR100(
            root=root, train=False, download=True, transform=test_transform
        )

    def _set_data_CIFAR10H(self, root, train_transform, test_transform):
        self.train_data = dsets.CIFAR10(
            root=root, train=True, download=True, transform=train_transform
        )

        self.test_data = dsets.CIFAR10(
            root=root, train=False, download=True, transform=test_transform
        )

        self.test_data.targets = list(zip(self.test_data.targets, probs))

    def _set_data_ImageNetO(self, root, train_transform, test_transform):
        self.train_data = EmptyDataset()
        self.test_data = get_imagnet_natural_adv(root, "ImageNet-O", test_transform)

    def _set_data_ImageNetA(self, root, train_transform, test_transform):
        self.train_data = EmptyDataset()
        self.test_data = get_imagnet_natural_adv(root, "ImageNet-A", test_transform)

    def _set_data_ImageNetR(self, root, train_transform, test_transform):
        self.train_data = EmptyDataset()
        self.test_data = get_imagnet_renditions(root, "ImageNet-R", test_transform)


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x_data = []
        self.y_data = []

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return _, _
