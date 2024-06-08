from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchSiLULayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSiLULayer, self).__init__()
        self.layer = nn.SiLU()
