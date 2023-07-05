from torch import nn

from layers.linear import LinearLayer


class PytorchLinear(LinearLayer):

    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.layer = self.__create_layer()

    def __create_layer(self):
        return nn.Linear(in_features=self.input_dimension, out_features=self.output_dimension)

    def forward(self, x):
        return self.layer(x)
