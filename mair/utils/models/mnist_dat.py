import torch
import torch.nn as nn

# https://github.com/YisenWang/dynamic_adv_training/blob/master/models.py
class MNIST_DAT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(  # 28
            nn.Conv2d(1, 32, 3, padding=1),  # , 28-3+1+2=28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14
            nn.Conv2d(32, 64, 3, padding=1),  # 14
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
