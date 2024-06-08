from typing import List

from framework.architecture.sections.recurrent import RecurrentSection
from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchRecurrentSection(PytorchSection, RecurrentSection):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchRecurrentSection, self).__init__(blocks)
