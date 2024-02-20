from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchReluLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchReluLayer, self).__init__()
        self.layer = nn.ReLU()
