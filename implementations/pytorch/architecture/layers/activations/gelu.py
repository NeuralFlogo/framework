from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchGELULayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchGELULayer, self).__init__()
        self.layer = nn.GELU()
