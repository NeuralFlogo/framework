from torch.nn import Module, Sequential


class PytorchNetwork(Module):
    def __init__(self, layers):
        super(PytorchNetwork, self).__init__()
        self.layers = layers
        self.architecture = Sequential()
        self.__build_network()

    def __build_network(self):
        [self.architecture.append(layer) for layer in self.layers]

    def forward(self, x):
        return self.architecture(x)
