from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchGLULayer(PytorchActivationLayer):
    def __init__(self):
        super(PytorchGLULayer, self).__init__()
        self.layer = nn.GLU()
