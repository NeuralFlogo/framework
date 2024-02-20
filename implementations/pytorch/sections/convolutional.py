from typing import List

from implementations.pytorch.blocks.convolutional import PytorchConvolutionalBlock
from implementations.pytorch.section import PytorchSection


class PytorchConvolutionalSection(PytorchSection):
    def __init__(self, blocks: List[PytorchConvolutionalBlock]):
        super(PytorchConvolutionalSection, self).__init__(blocks)
