from typing import List

from framework.architecture.block import Block
from framework.architecture.section import Section


class ResidualSection(Section):
    def __init__(self, blocks: List[Block]):
        super().__init__(blocks)
