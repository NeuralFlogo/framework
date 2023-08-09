from torch import nn

from framework.structure.layers.activation import ActivationLayer


class PytorchLogSigmoid(nn.Module, ActivationLayer):
    def __init__(self):
        nn.Module.__init__(self)
        self.layer = nn.LogSigmoid()

    def forward(self, x):
        return self.layer(x)
