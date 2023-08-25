from blocks.residual import ResidualBlock
from layers.pool import PoolingLayer
from section import Section


class ResidualSection(Section):
    def __init__(self, residual_block: ResidualBlock, pooling: PoolingLayer):
        self.residual_block = residual_block
        self.pooling = pooling

    def layers(self):
        return *self.residual_block.layers(), self.pooling.get()
