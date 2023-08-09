from framework.structure.blocks.linear import LinearBlock
from section import Section


class LinearSection(Section):
    def __init__(self, blocks: list[LinearBlock]):
        self.blocks = blocks

    def layers(self):
        layers = []
        for block in self.blocks:
            layers += block.layers()
        return layers
        # return [block.layers() for block in self.blocks]
