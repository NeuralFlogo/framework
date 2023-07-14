from blocks.convolutional import ConvolutionalBlock
from generics.section import Section
from layers.pool import PoolingLayer


class ConvolutionalSection(Section):
    def __init__(self, blocks: list[ConvolutionalBlock], pooling: PoolingLayer):
        super().__init__(blocks)
        self.pooling = pooling

    def get_layers(self):
        return super(ConvolutionalSection, self).get_layers() + [self.pooling]
