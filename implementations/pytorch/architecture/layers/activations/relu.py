from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchReLULayer(PytorchActivationLayer):
    def __init__(self, inplace: bool = False):
        super(PytorchReLULayer, self).__init__()
        self.layer = nn.ReLU(inplace=inplace)
