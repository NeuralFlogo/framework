from torch import nn

from layers.activation import ActivationLayer


class PytorchSilu(ActivationLayer):
    def __init__(self):
        self.layer = nn.SiLU()

    def get(self):
        return self.layer
