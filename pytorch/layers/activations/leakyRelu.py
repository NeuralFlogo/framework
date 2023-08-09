from torch import nn

from framework.structure.layers.activation import ActivationLayer


class PytorchLeakyRelu(nn.Module, ActivationLayer):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer = nn.LeakyReLU()

    def forward(self, x):
        return self.layer(x)
