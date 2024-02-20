from typing import List

from implementations.pytorch.blocks.recurrent import PytorchRecurrentBlock
from implementations.pytorch.section import PytorchSection


class PytorchRecurrentSection(PytorchSection):
    def __init__(self, blocks: List[PytorchRecurrentBlock]):
        super(PytorchRecurrentSection, self).__init__(blocks)
