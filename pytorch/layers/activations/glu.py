from torch import nn

from layers.activation import ActivationLayer


class PytorchGlu(ActivationLayer):
    def __init__(self):
        self.layer = nn.GLU()

    def get(self):
        return self.layer
