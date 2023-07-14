from blocks.linear import LinearBlock
from generics.section import Section


class LinearSection(Section):
    def __init__(self, blocks: list[LinearBlock]):
        super().__init__(blocks)
