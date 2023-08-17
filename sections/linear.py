from blocks.linear import LinearBlock
from section import Section


class LinearSection(Section):
    def __init__(self, blocks: list[LinearBlock]):
        self.blocks = blocks

    def layers(self):
        return tuple(layer for block in self.blocks for layer in block.layers())
