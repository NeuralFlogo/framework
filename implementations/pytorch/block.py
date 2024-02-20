from typing import List

from torch import nn

from framework.block import Block
from implementations.pytorch.layer import PytorchLayer


class PytorchBlock(Block, nn.Module):
    def __init__(self, layers: List[PytorchLayer]):
        super(PytorchBlock, self).__init__(layers)
