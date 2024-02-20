from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchSiluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSiluLayer, self).__init__()
        self.layer = nn.SiLU()
