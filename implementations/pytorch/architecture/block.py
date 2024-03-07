from typing import List

from torch import Tensor
from torch.nn import Module, Sequential

from framework.architecture.block import Block
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchBlock(Module, Block):
    def __init__(self, layers: List[PytorchLayer]):
        Module.__init__(self)
        Block.__init__(self, layers)
        self.block = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
