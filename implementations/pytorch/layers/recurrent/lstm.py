from torch import nn

from implementations.pytorch.layers.recurrent import PytorchRecurrentLayer


class PytorchLSTMLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int):
        super(PytorchLSTMLayer, self).__init__()
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer)
