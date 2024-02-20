from torch import nn

from framework.layers.linear import LinearLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchLinearLayer(PytorchLayer, LinearLayer):
    def __init__(self, in_features: int, out_features: int):
        super(PytorchLinearLayer, self).__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features)
