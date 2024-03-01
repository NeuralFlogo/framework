from torch import nn

from implementations.pytorch.architecture.layers.activation import PytorchActivationLayer


class PytorchSoftmaxLayer(PytorchActivationLayer):
    def __init__(self, n_dimensions: int):
        super(PytorchSoftmaxLayer, self).__init__()
        self.layer = nn.Softmax(dim=n_dimensions)
