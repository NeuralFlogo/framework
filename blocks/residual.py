from block import Block
from layers.residual import ResidualLayer


class ResidualBlock(Block):
    def __init__(self, residuals: list[ResidualLayer]):
        self.residuals = residuals

    def layers(self):
        return tuple(layer.get() for layer in self.residuals)
