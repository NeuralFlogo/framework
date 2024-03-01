from typing import List

from implementations.pytorch.architecture.blocks.residual import PytorchResidualBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchResidualSection(PytorchSection):
    def __init__(self, blocks: List[PytorchResidualBlock]):
        super(PytorchResidualSection, self).__init__(blocks)
