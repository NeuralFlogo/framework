from torch import nn

from framework.architecture.layers.recurrent import RecurrentLayer
from implementations.pytorch.architecture.layers.recurrent import PytorchRecurrentLayer


class PytorchRNNLayer(PytorchRecurrentLayer):
    def __init__(self, input_size: int, hidden_size: int, output_type: RecurrentLayer.OutputType, num_layer: int, bidirectional: bool, dropout: float):
        super(PytorchRNNLayer, self).__init__(output_type)
        self.layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, bidirectional=bidirectional, dropout=dropout, batch_first=True)
