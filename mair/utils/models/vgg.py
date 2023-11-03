import torch
import torch.nn as nn

model_info = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _make_layers(model_name):
    layers = []
    in_channel = 3

    for out_channel in model_info[model_name]:
        if out_channel == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            in_channel = out_channel

    return layers


class VGG(nn.Module):
    def __init__(self, name, num_classes=10):
        super(VGG, self).__init__()

        self.conv_layer = nn.Sequential(*_make_layers(name))

        self.fc_layer = nn.Sequential(nn.Linear(512, num_classes))

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(-1, 512)
        out = self.fc_layer(out)

        return out
