from blocks.convolutional import ConvolutionalBlock
from section import Section
from layers.pool import PoolingLayer


class ConvolutionalSection(Section):
    def __init__(self, blocks: list[ConvolutionalBlock], pooling: PoolingLayer):
        self.blocks = blocks
        self.pooling = pooling

    def layers(self):
        return [*self.unstack_blocks(), self.pooling]

    def unstack_blocks(self):
        return [block.layers() for block in self.blocks]
