from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchLogSigmoidLayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchLogSigmoidLayer, self).__init__()
        self.layer = nn.LogSigmoid()
