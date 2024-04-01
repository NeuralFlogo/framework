from typing import List

from framework.architecture.sections.linear import LinearSection
from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchLinearSection(PytorchSection, LinearSection):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchLinearSection, self).__init__(blocks)
