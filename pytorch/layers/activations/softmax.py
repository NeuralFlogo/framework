from torch import nn

from layers.activation import ActivationLayer


class PytorchSoftmax(ActivationLayer):
    def __init__(self, n_dimensions):
        self.layer = nn.Softmax(n_dimensions)

    def get(self):
        return self.layer
