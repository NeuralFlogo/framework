from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchLeakyReLULayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchLeakyReLULayer, self).__init__()
        self.layer = nn.LeakyReLU()
