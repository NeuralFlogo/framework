from torch import nn
from framework.structure.layers.flatten import FlattenLayer


class PytorchFlatten(FlattenLayer, nn.Module):
    def __init__(self, start_dim, end_dim):
        nn.Module.__init__(self)
        FlattenLayer.__init__(self, start_dim, end_dim)
        self.layer = nn.Flatten(start_dim=self.start_dim, end_dim=self.end_dim)

    def forward(self, x):
        return self.layer(x)
