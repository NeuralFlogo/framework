from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchGluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchGluLayer, self).__init__()
        self.layer = nn.GLU()
