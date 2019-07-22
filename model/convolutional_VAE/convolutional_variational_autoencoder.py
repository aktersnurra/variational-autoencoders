import torch.nn as nn

from utils import initialize_weights
from utils.gelu import GELU


class InferenceNetwork(nn.Module):
    """

    """

    def __init__(self, params):
        super(InferenceNetwork).__init__()

        conv_layers = [
            nn.Conv2d(in_channels=params.channels[0],
                      out_channels=params.channels[1],
                      kernel_size=params.kernel_size[0],
                      stride=params.stride[0],
                      padding=params.padding[0]),
            nn.BatchNorm2d(params.channels[1]),
            GELU(),
            nn.Conv2d(in_channels=params.channels[1],
                      out_channels=params.channels[2],
                      kernel_size=params.kernel_size[1],
                      stride=params.stride[1]),
            nn.BatchNorm2d(params.channels[2]),
            GELU(),
            nn.Conv2d(in_channels=params.channels[2],
                      out_channels=params.channels[3],
                      kernel_size=params.kernel_size[2],
                      stride=params.stride[2]),
            nn.BatchNorm2d(params.channels[3]),
            GELU(),
        ]

        self.encoding = nn.Sequential(*conv_layers)
        initialize_weights(self)

    def forward(self, x):
        return self.encoding(x)


class GenerativeNetwork(nn.Module):
    def __init__(self, params):
        super(GenerativeNetwork).__init__()
        conv_layers = [
            nn.ConvTranspose2d(in_channels=params.channels[0],
                               out_channels=params.channels[1],
                               kernel_size=params.kernel_size[0],
                               stride=params.stride[0]),
            nn.BatchNorm2d(params.channels[1]),
            GELU(),
            nn.ConvTranspose2d(in_channels=params.channels[1],
                               out_channels=params.channels[2],
                               kernel_size=params.kernel_size[1],
                               stride=params.stride[1]),
            nn.BatchNorm2d(params.channels[2]),
            GELU(),
            nn.ConvTranspose2d(in_channels=params.channels[2],
                               out_channels=params.channels[3],
                               kernel_size=params.kernel_size[2],
                               stride=params.stride[2]),
            nn.BatchNorm2d(params.channels[3]),
            GELU(),
        ]

        self.decoding = nn.Sequential(*conv_layers)
        initialize_weights(self)

    def forward(self, z):
        return self.decoding(z)


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
