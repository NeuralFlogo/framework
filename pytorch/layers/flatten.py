from torch import nn

from layers.flatten import FlattenLayer


class PytorchFlatten(FlattenLayer):
    def __init__(self, start_dim, end_dim):
        self.layer = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def get(self):
        return self.layer
