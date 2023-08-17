from torch import nn

from layers.activation import ActivationLayer


class PytorchRelu(ActivationLayer):
    def __init__(self):
        self.layer = nn.ReLU()

    def get(self):
        return self.layer
