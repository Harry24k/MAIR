import torch
import torch.nn as nn

# Fast is better than free (https://arxiv.org/abs/2001.03994)
class MNIST_Fast(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_Fast, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
