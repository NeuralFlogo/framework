from torch import nn

from framework.structure.layers.linear import LinearLayer


class PytorchLinear(nn.Module, LinearLayer):
    def __init__(self, in_dimension, out_dimension):
        nn.Module.__init__(self)
        LinearLayer.__init__(self, in_dimension, out_dimension)
        self.layer = nn.Linear(in_features=self.in_dimension, out_features=self.out_dimension)

    def forward(self, x):
        return self.layer(x)
