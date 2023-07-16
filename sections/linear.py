from blocks.linear import LinearBlock
from section import Section


class LinearSection(Section):
    def __init__(self, blocks: list[LinearBlock]):
        self.blocks = blocks

    def get_layers(self):
        return [block.get_layers() for block in self.blocks]
