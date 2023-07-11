from torch import nn
from layers.flatten import FlattenLayer


class PytorchFlatten(FlattenLayer):
    def __init__(self, start_dim, end_dim):
        super().__init__(start_dim, end_dim)
        self.layer = nn.Flatten(start_dim=self.start_dim, end_dim=self.end_dim)

    def forward(self, x):
        return self.layer(x)
