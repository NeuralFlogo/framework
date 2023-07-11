from torch import nn

from layers.activation import ActivationLayer


class PytorchRelu(ActivationLayer):

    def __init__(self):
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.ReLU()

    def forward(self, x):
        return self.layer(x)