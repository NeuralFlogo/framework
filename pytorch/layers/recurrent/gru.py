import torch
from torch import nn

from layers.recurrent import RecurrentLayer


class PytorchGRU(RecurrentLayer):
    def __init__(self, input_size, hidden_size, num_layer):
        self.layer = self.__build_block(input_size, hidden_size, num_layer)

    def __build_block(self, input_size, hidden_size, num_layer):
        return self.Block(input_size, hidden_size, num_layer)

    def get(self):
        return self.layer

    class Block(nn.Module):
        def __init__(self, input_size, hidden_size, num_layer):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layer = num_layer
            self.layer = self.__create_layer()
            self.hidden_state = self.__init_hidden_state()

        def __create_layer(self):
            return nn.GRU(self.input_size, self.hidden_size, self.num_layer, batch_first=True)

        def __init_hidden_state(self):
            return torch.zeros(self.num_layer, self.input_size, self.hidden_size)

        def forward(self, x):
            x, self.hidden_state = self.layer(x, self.hidden_state)
