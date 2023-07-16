from torch import nn

from layers.activation import ActivationLayer


class PytorchSoftMax(nn.Module, ActivationLayer):
    def __init__(self, n_dimensions):
        nn.Module.__init__(self)
        self.n_dimensions = n_dimensions
        self.layer = nn.Softmax(self.n_dimensions)

    def forward(self, x):
        return self.layer(x)
