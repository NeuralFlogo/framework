from torch import nn

from layers.activation import ActivationLayer


class PytorchSelu(ActivationLayer):
    def __init__(self):
        self.layer = nn.SELU()

    def get(self):
        return self.layer
