from block import Block
from layers.activation import ActivationLayer
from layers.convolutional import ConvolutionalLayer
from layers.normalization import NormalizationLayer


class ConvolutionalBlock(Block):
    def __init__(self, convolution: ConvolutionalLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        self.convolution = convolution
        self.activation = activation
        self.normalization = normalization

    def layers(self):
        return self.convolution, self.activation, self.normalization
