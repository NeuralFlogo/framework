from torch import nn

from framework.architecture.layers.linear import LinearLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchLinearLayer(PytorchLayer, LinearLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PytorchLinearLayer, self).__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
