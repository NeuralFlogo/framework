from typing import List

from torch import Tensor
from torch.nn import Module, Sequential

from framework.architecture.blocks.residual import ResidualBlock
from implementations.pytorch.architecture.layer import PytorchLayer


class PytorchResidualBlock(Module, ResidualBlock):
    def __init__(self, layers: List[PytorchLayer], shortcut: List[PytorchLayer]):
        Module.__init__(self, layers)
        ResidualBlock.__init__(self, layers, shortcut)
        self.block = Sequential(*layers)
        self.shortcut = Sequential(*shortcut)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.shortcut(x)
