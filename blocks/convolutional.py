from generics.block import Block
from layers.activation import ActivationLayer
from layers.convolutional import ConvolutionalLayer
from layers.normalization import NormalizationLayer


class ConvolutionalBlock(Block):
    def __init__(self, convolution: ConvolutionalLayer, activation: ActivationLayer = None,
                 normalization: NormalizationLayer = None):
        super().__init__([layer for layer in (convolution, activation, normalization) if layer is not None])
