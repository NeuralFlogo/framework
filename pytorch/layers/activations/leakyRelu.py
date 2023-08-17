from torch import nn

from layers.activation import ActivationLayer


class PytorchLeakyRelu(ActivationLayer):
    def __init__(self):
        self.layer = nn.LeakyReLU()

    def get(self):
        return self.layer
