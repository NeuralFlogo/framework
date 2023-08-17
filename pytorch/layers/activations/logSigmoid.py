from torch import nn

from layers.activation import ActivationLayer


class PytorchLogSigmoid(ActivationLayer):
    def __init__(self):
        self.layer = nn.LogSigmoid()

    def get(self):
        return self.layer
