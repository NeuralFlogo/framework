from typing import List

from implementations.pytorch.architecture.blocks.recurrent import PytorchRecurrentBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchRecurrentSection(PytorchSection):
    def __init__(self, blocks: List[PytorchRecurrentBlock]):
        super(PytorchRecurrentSection, self).__init__(blocks)
