from typing import Tuple

from torch import nn, Tensor
from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer


class PytorchLSTMLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, num_layer: int, bidirectional: bool, dropout: float):
        super(PytorchLSTMLayer, self).__init__()
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor | Tuple[Tensor]:
        output, (hidden, cell) = self.layer(x)
        return output, hidden, cell
