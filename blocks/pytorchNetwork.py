from torch import nn

from network import Network


class PytorchNetwork(nn.Module):

    def __init__(self, layers):
        self.layers = layers
        super(PytorchNetwork, self).__init__()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

