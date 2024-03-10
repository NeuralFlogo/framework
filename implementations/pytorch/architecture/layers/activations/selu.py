from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchSELULayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSELULayer, self).__init__()
        self.layer = nn.SELU()
