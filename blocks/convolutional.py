from block import Block
from layers.activation import ActivationLayer
from layers.convolutional import ConvolutionalLayer
from layers.normalization import NormalizationLayer


class ConvolutionalBlock(Block):
    def __init__(self, convolution: ConvolutionalLayer, pre_normalization: NormalizationLayer = None, activation: ActivationLayer = None, post_normalization: NormalizationLayer = None):
        self.convolution = convolution
        self.pre_normalization = pre_normalization
        self.activation = activation
        self.post_normalization = post_normalization

    def layers(self):
        block = [self.convolution.get()]
        if self.pre_normalization: block.append(self.pre_normalization.get())
        if self.activation: block.append(self.activation.get())
        if self.post_normalization: block.append(self.post_normalization.get())
        return tuple(block)
