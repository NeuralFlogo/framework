from torch import nn

from framework.architecture.layers.flatten import FlattenLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchFlattenLayer(PytorchLayer, FlattenLayer):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super(PytorchFlattenLayer, self).__init__()
        self.layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
