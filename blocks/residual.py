from block import Block
from layers.residual import ResidualLayer


class ResidualBlock(Block):
    def __init__(self, residuals: list[ResidualLayer]):
        self.residuals = residuals

    def layers(self):
        return tuple(residual.get() for residual in self.residuals)
