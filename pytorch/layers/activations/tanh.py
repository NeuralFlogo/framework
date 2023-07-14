from torch import nn

from layers.activation import ActivationLayer


class PytorchTanh(nn.Module, ActivationLayer):

    def __init__(self):
        nn.Module.__init__(self)
        self.layer = nn.Tanh()

    def forward(self, x):
        return self.layer(x)
