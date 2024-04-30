from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchELULayer(PytorchActivationLayer):
    def __init__(self, alpha: float = 1.0):
        super(PytorchELULayer, self).__init__()
        self.layer = nn.ELU(alpha=alpha)
