from block import Block
from framework.structure.layers.activation import ActivationLayer
from framework.structure.layers.linear import LinearLayer
from framework.structure.layers.normalization import NormalizationLayer


class LinearBlock(Block):
    def __init__(self, linear: LinearLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        self.linear = linear
        self.activation = activation
        self.normalization = normalization

    def layers(self):

        return [self.linear, self.activation, self.normalization]
