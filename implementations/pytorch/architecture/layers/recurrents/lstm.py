from typing import Tuple

from torch import nn, Tensor

from framework.architecture.layers.recurrent import RecurrentLayer
from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer


class PytorchLSTMLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, output_type: RecurrentLayer.OutputType, num_layer: int, bidirectional: bool, dropout: float):
        super(PytorchLSTMLayer, self).__init__(output_type)
        self.layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        end_sequence, (hidden, cell) = self.layer(x)
        output = (end_sequence, hidden.transpose(0, 1), cell.transpose(0, 1))
        return output[self.output.value]
