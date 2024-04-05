from torch import nn

from framework.architecture.layers.flatten import FlattenLayer
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchFlattenLayer(PytorchLayer, FlattenLayer):
    def __init__(self, from_dim: int = 1, to_dim: int = -1):
        super(PytorchFlattenLayer, self).__init__()
        self.layer = nn.Flatten(start_dim=from_dim, end_dim=to_dim)
