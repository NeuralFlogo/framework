from torch import nn
from layers.linear import LinearLayer


class PytorchLinear(LinearLayer):
    def __init__(self, input_dimension, output_dimension):
        super().__init__(input_dimension, output_dimension)
        self.layer = nn.Linear(in_features=self.in_dimension, out_features=self.out_dimension)

    def forward(self, x):
        return self.layer(x)
