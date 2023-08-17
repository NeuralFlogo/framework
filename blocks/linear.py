from block import Block
from layers.activation import ActivationLayer
from layers.linear import LinearLayer
from layers.normalization import NormalizationLayer


class LinearBlock(Block):
    def __init__(self, linear: LinearLayer, prev_normalizations: list[NormalizationLayer] = None, activation: ActivationLayer = None, post_normalizations: list[NormalizationLayer] = None):
        self.linear = linear
        self.prev_normalizations = prev_normalizations
        self.activation = activation
        self.post_normalizations = post_normalizations

    def layers(self):
        block = [self.linear.get()]
        if self.prev_normalizations: self.__append_bacth_of_layers(self.prev_normalizations, block)
        if self.activation: block.append(self.activation.get())
        if self.post_normalizations: self.__append_bacth_of_layers(self.post_normalizations, block)
        return tuple(block)

    def __append_bacth_of_layers(self, batch, block):
        for layer in batch:
            block.append(layer.get())
        return block
