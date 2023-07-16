from block import Block
from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.normalization import NormalizationLayer


class LinearBlock(Block):
    def __init__(self, linear: LinearLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        self.linear = linear
        self.activation = activation
        self.normalization = normalization

    def get_layers(self):
        return self.linear, self.activation, self.normalization
