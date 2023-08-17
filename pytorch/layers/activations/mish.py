from torch import nn

from layers.activation import ActivationLayer


class PytorchMish(ActivationLayer):
    def __init__(self):
        self.layer = nn.Mish()

    def get(self):
        return self.layer
