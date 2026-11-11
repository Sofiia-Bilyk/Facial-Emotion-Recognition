from torch import nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


def build_model():
    return nn.Sequential(
        conv_block(1, 32),
        conv_block(32, 64),
        conv_block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(start_dim=1),
        nn.Dropout(p=0.3),
        nn.Linear(128, 7),
    )
