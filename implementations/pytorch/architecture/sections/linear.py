from typing import List

from implementations.pytorch.architecture.blocks.linear import PytorchLinearBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchLinearSection(PytorchSection):
    def __init__(self, blocks: List[PytorchLinearBlock]):
        super(PytorchLinearSection, self).__init__(blocks)
