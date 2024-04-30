from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchLeakyReluLayer(PytorchActivationLayer):
    def __init__(self, negative_slope: float = 0.01):
        super(PytorchLeakyReluLayer, self).__init__()
        self.layer = nn.LeakyReLU(negative_slope=negative_slope)
