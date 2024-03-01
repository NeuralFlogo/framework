import torch
from torch import nn

from framework.architecture.layer import Layer


class PytorchLayer(Layer, nn.Module):
    def __init__(self):
        super(PytorchLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
