from typing import List

from torch import Tensor
from torch.nn import Module


from framework.architecture.section import Section
from implementations.pytorch.architecture.block import PytorchBlock


class PytorchSection(Module, Section):
    def __init__(self, blocks: List[PytorchBlock]):
        Module.__init__(self)
        Section.__init__(self, blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x
