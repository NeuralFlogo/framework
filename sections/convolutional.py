from blocks.convolutional import ConvolutionalBlock
from section import Section
from layers.pool import PoolingLayer


class ConvolutionalSection(Section):
    def __init__(self, blocks: list[ConvolutionalBlock], pooling: PoolingLayer):
        self.blocks = blocks
        self.pooling = pooling

    def layers(self):
        layers = self.unstack_layers_from_blocks().append(self.pooling)
        return layers

    def unstack_layers_from_blocks(self):
        return [block.layers() for block in self.blocks]
