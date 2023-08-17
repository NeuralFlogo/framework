from torch import nn

from layers.activation import ActivationLayer


class PytorchGelu(ActivationLayer):
    def __init__(self):
        self.layer = nn.GELU()

    def get(self):
        return self.layer
