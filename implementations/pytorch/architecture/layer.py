import torch
from torch import nn, Tensor

from framework.architecture.layer import Layer


class PytorchLayer(Layer, nn.Module):
    def __init__(self):
        super(PytorchLayer, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

    def params(self):
        print(self.parameters())
