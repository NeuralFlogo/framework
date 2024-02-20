from typing import List

from framework.blocks.linear import LinearBlock
from framework.section import Section


class LinearSection(Section):
    def __init__(self, blocks: List[LinearBlock]):
        super().__init__(blocks)
