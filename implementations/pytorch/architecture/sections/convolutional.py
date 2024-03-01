from typing import List

from implementations.pytorch.architecture.blocks.convolutional import PytorchConvolutionalBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchConvolutionalSection(PytorchSection):
    def __init__(self, blocks: List[PytorchConvolutionalBlock]):
        super(PytorchConvolutionalSection, self).__init__(blocks)
