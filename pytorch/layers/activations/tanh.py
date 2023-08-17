from torch import nn

from layers.activation import ActivationLayer


class PytorchTanh(ActivationLayer):
    def __init__(self):
        self.layer = nn.Tanh()

    def get(self):
        return self.layer
