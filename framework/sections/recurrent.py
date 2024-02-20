from typing import List

from framework.blocks.recurrent import RecurrentBlock
from framework.section import Section


class RecurrentSection(Section):
    def __init__(self, blocks: List[RecurrentBlock]):
        super().__init__(blocks)
