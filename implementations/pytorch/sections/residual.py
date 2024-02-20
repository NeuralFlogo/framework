from typing import List

from implementations.pytorch.blocks.residual import PytorchResidualBlock
from implementations.pytorch.section import PytorchSection


class PytorchResidualSection(PytorchSection):
    def __init__(self, blocks: List[PytorchResidualBlock]):
        super(PytorchResidualSection, self).__init__(blocks)
