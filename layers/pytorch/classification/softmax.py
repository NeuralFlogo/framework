from torch import nn

from layers.classification import ClassificationLayer


class PytorchSoftMax(ClassificationLayer):

    def __init__(self, dimensions):
        super().__init__(dimensions)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.Softmax(dim=self.dimensions)

    def forward(self, x):
        return self.layer(x)