import torch
from torch import nn, Tensor

from framework.architecture.layers.linear import LinearLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchLinearLayer(PytorchLayer, LinearLayer):
    def __init__(self, in_features: int, out_features: int, dimension: int = -1, bias: bool = True):
        super(PytorchLinearLayer, self).__init__()
        self.dimension = dimension
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.dimension == -1:
            return self.layer(x)
        return torch.transpose(self.layer(torch.transpose(x, self.dimension, -1)), self.dimension, -1)
