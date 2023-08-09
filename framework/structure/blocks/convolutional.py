from block import Block
from framework.structure.layers.activation import ActivationLayer
from framework.structure.layers.convolutional import ConvolutionalLayer
from framework.structure.layers.normalization import NormalizationLayer


class ConvolutionalBlock(Block):
    def __init__(self, convolution: ConvolutionalLayer, activation: ActivationLayer = None, normalization: NormalizationLayer = None):
        self.convolution = convolution
        self.activation = activation
        self.normalization = normalization

    def layers(self):
        return [self.convolution, self.activation, self.normalization]
