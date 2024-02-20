from torch import nn

from implementations.pytorch.layers.recurrent import PytorchRecurrentLayer


class PytorchRNNLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int):
        super(PytorchRNNLayer, self).__init__()
        self.layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer)
