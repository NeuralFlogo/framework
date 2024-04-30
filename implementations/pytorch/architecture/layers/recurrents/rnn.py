from torch import nn

from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer


class PytorchRnnLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int, bidirectional: bool, dropout: float):
        super(PytorchRnnLayer, self).__init__()
        self.layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout)
