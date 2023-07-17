from torch.nn import Module, Sequential


class PytorchNetwork(Module): # TODO how will this work when there are multiple inputs?
    def __init__(self):
        super(PytorchNetwork, self).__init__()
        self.architecture = Sequential()

    def build(self):
        [self.architecture.append(layer) for layer in self.layers]

    def forward(self, x):
        return self.architecture(x)
