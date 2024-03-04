from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchReluLayer(PytorchActivationLayer):
    def __init__(self, inplace: bool = False):
        super(PytorchReluLayer, self).__init__()
        self.layer = nn.ReLU(inplace=inplace)
