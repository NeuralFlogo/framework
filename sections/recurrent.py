from blocks.recurrent import RecurrentBlock
from section import Section


class RecurrentSection(Section):
    def __init__(self, block: RecurrentBlock):
        self.block = block

    def layers(self):
        return self.block.layers()

