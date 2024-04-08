from torch import nn

from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer


class PytorchGRULayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int, bidirectional: bool, dropout: float):
        super(PytorchGRULayer, self).__init__()
        self.layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout)
