from block import Block
from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.normalization import NormalizationLayer


class LinearBlock(Block):
    def __init__(self, linear: LinearLayer, pre_normalization: NormalizationLayer = None, activation: ActivationLayer = None, post_normalization: NormalizationLayer = None):
        self.linear = linear
        self.pre_normalization = pre_normalization
        self.activation = activation
        self.post_normalization = post_normalization

    def layers(self):
        block = [self.linear.get()]
        if self.pre_normalization: block.append(self.pre_normalization.get())
        if self.activation: block.append(self.activation.get())
        if self.post_normalization: block.append(self.post_normalization.get())
        return tuple(block)
