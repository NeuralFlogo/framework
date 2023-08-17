from torch import nn

from layers.activation import ActivationLayer


class PytorchSigmoid(ActivationLayer):
    def __init__(self):
        self.layer = nn.Sigmoid()

    def get(self):
        return self.layer
