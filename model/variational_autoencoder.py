import torch.nn as nn
import torch.nn.functional as F
from ..utils import initialize_weights


class ContextNetwork(nn.Module):
    def __init__(self, params):

        super(ContextNetwork).__init__()

        layers = [
            nn.Conv2d(in_channels=params.channels[0],
                      out_channels=params.channels[1],
                      kernel_size=params.kernel_size[0],
                      stride=params.stride[0]),
            nn.BatchNorm2d(params.channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params.channels[1],
                      out_channels=params.channels[2],
                      kernel_size=params.kernel_size[1],
                      stride=params.stride[1]),
            nn.BatchNorm2d(params.channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=params.channels[2],
                      out_channels=params.channels[3],
                      kernel_size=params.kernel_size[2],
                      stride=params.stride[2]),
            nn.BatchNorm2d(params.channels[3]),
            nn.ReLU(inplace=True),
        ]

        self.context_image = nn.Sequential(*layers)
        initialize_weights(self)

    def forward(self, img_coarse):
        return self.context_image(img_coarse)
