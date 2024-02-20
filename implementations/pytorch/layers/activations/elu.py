from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchEluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchEluLayer, self).__init__()
        self.layer = nn.ELU()
