from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchLeakyReLULayer(PytorchActivationLayer):
    def __init__(self, negative_slope: float = 0.01):
        super(PytorchLeakyReLULayer, self).__init__()
        self.layer = nn.LeakyReLU(negative_slope=negative_slope)
