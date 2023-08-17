from blocks.convolutional import ConvolutionalBlock
from section import Section
from layers.pool import PoolingLayer


class ConvolutionalSection(Section):
    def __init__(self, blocks: list[ConvolutionalBlock], pooling: PoolingLayer):
        self.blocks = blocks
        self.pooling = pooling

    def layers(self):
        layers = self.unstack_layers_from_blocks()
        layers.append(self.pooling.get())
        return tuple(layers)

    def unstack_layers_from_blocks(self):
        return [layer for block in self.blocks for layer in block.layers()]
