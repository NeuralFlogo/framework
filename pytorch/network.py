from torch.nn import Module, Sequential

from architecture import Architecture


class PytorchNetwork(Architecture, Module):
    def __init__(self, layers):
        super(Architecture).__init__(layers)
        super(Module).__init__()
        self.architecture = Sequential()
        self.__build_network()

    def __build_network(self):
        [self.architecture.append(layer) for layer in self.layers]

    def forward(self, x):
        return self.architecture(x)
