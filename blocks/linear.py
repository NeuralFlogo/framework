from generics.block import Block
from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.normalization import NormalizationLayer


class LinearBlock(Block):
    def __init__(self, linear: LinearLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        super().__init__([layer for layer in (linear, activation, normalization) if layer is not None])
