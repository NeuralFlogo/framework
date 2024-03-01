from typing import List

from framework.architecture.blocks.linear import LinearBlock
from framework.architecture.section import Section


class LinearSection(Section):
    def __init__(self, blocks: List[LinearBlock]):
        super().__init__(blocks)
