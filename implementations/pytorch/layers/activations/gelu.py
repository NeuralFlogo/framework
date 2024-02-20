from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchGeluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchGeluLayer, self).__init__()
        self.layer = nn.GELU()
