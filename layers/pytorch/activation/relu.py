from torch import nn


class PytorchRelu:

    def __init__(self):
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.ReLU()

    def forward(self, x):
        return self.layer(x)