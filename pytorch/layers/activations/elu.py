from torch import nn

from layers.activation import ActivationLayer


class PytorchElu(ActivationLayer):
    def __init__(self):
        self.layer = nn.ELU()

    def get(self):
        return self.layer
