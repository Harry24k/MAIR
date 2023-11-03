# Modified from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert len(mean) == len(std)
        self.n_channels = len(mean)
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def __str__(self):
        return "Normalize(" + "mean={}, std={}".format(self.mean, self.std) + ")"

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def forward(self, inputs):
        # Broadcasting
        mean = self.mean.reshape(1, self.n_channels, 1, 1).to(inputs.device)
        std = self.std.reshape(1, self.n_channels, 1, 1).to(inputs.device)
        return (inputs - mean) / std


class InverseNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert len(mean) == len(std)
        self.n_channels = len(mean)
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def __str__(self):
        return "InverseNormalize(" + "mean={}, std={}".format(self.mean, self.std) + ")"

    def extra_repr(self):
        return "mean={}, std={}".format(self.mean, self.std)

    def forward(self, inputs):
        # Broadcasting
        mean = self.mean.reshape(1, self.n_channels, 1, 1).to(inputs.device)
        std = self.std.reshape(1, self.n_channels, 1, 1).to(inputs.device)
        return (inputs * std) + mean
