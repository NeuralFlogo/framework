from blocks.convolutional import ConvolutionalBlock
from layers.pool import PoolingLayer


class ConvolutionalSection:
    def __init__(self, blocks: list[ConvolutionalBlock], pooling: PoolingLayer):
        self.blocks = blocks
        self.pooling = pooling
