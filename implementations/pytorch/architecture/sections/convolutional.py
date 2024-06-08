from typing import List

from framework.architecture.sections.convolutional import ConvolutionalSection
from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.section import PytorchSection


class PytorchConvolutionalSection(PytorchSection, ConvolutionalSection):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchConvolutionalSection, self).__init__(blocks)
