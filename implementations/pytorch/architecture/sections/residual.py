from typing import List

from framework.architecture.sections.residual import ResidualSection
from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchResidualSection(PytorchSection, ResidualSection):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchResidualSection, self).__init__(blocks)
