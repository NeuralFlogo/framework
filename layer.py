from torch import nn


class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()
        self.layer

    def forward(self, x):
        return self.layer(x)
