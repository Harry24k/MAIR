# Modified from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/wrn.py
import torch
import torch.nn as nn


class Identity(nn.Module):
    def forward(self, inputs):
        return inputs
