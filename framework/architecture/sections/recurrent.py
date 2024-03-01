from typing import List

from framework.architecture.blocks.recurrent import RecurrentBlock
from framework.architecture.section import Section


class RecurrentSection(Section):
    def __init__(self, blocks: List[RecurrentBlock]):
        super().__init__(blocks)
