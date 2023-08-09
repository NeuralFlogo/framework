from torch.nn import Module, Sequential

from dataset_adder import DatasetAdder


class PytorchNetwork(Module):
    def __init__(self):
        super(PytorchNetwork, self).__init__()
        self.architectures = []

    def build(self, layers):
        sequential = Sequential()
        for layer in layers:
            if isinstance(layer, DatasetAdder):
                sequential = self.__new_add_layer(layer, sequential)
            else:
                sequential.append(layer)
        return self

    def __new_add_layer(self, layer, sequential):
        self.architectures.append(sequential)
        self.architectures.append(layer)
        sequential = Sequential()
        return sequential

    def forward(self, x):
        for architecture in self.architectures:
            x = architecture(x)
        return x
