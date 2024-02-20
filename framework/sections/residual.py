from framework.blocks.residual import ResidualBlock
from typing import List
from framework.section import Section


class ResidualSection(Section):
    def __init__(self, blocks: List[ResidualBlock]):
        super().__init__(blocks)
