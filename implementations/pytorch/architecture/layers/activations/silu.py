from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchSiluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSiluLayer, self).__init__()
        self.layer = nn.SiLU()
