from torch import nn

from framework.layers.flatten import FlattenLayer
from implementations.pytorch.layer import PytorchLayer


class PytorchFlatten(PytorchLayer, FlattenLayer):
    def __init__(self, start_dim: int, end_dim: int):
        super(PytorchFlatten, self).__init__()
        self.layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
