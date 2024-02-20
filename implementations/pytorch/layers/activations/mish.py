from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchMishLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchMishLayer, self).__init__()
        self.layer = nn.Mish()
