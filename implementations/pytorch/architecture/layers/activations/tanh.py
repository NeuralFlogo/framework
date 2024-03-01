from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchTanhLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchTanhLayer, self).__init__()
        self.layer = nn.Tanh()
