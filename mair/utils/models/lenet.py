import torch
import torch.nn as nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/models.py
class LeNetPP(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNetPP, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return y
