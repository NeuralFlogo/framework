from typing import List

from framework.architecture.block import Block
from framework.architecture.layer import Layer


class ResidualBlock(Block):
    def __init__(self, layers: List[Layer], shortcut: List[Layer]):
        super(ResidualBlock, self).__init__(layers)
        self.layers = layers
        self.shortcut = shortcut
