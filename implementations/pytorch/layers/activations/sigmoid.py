from torch import nn

from implementations.pytorch.layers.activation import PytorchActivationLayer


class PytorchSigmoidLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchSigmoidLayer, self).__init__()
        self.layer = nn.Sigmoid()
