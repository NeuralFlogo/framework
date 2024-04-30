from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchSeluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSeluLayer, self).__init__()
        self.layer = nn.SELU()
