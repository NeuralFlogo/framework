from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchLeakyReluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchLeakyReluLayer, self).__init__()
        self.layer = nn.LeakyReLU()
