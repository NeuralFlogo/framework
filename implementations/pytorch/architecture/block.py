from typing import List

from torch import Tensor
from torch.nn import Module

from framework.architecture.block import Block
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchBlock(Module, Block):
    def __init__(self, layers: List[PytorchLayer]):
        Module.__init__(self)
        Block.__init__(self, layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
