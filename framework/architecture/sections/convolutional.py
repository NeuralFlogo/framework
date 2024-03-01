from typing import List

from framework.architecture.blocks.convolutional import ConvolutionalBlock
from framework.architecture.section import Section


class ConvolutionalSection(Section):
    def __init__(self, blocks: List[ConvolutionalBlock]):
        super().__init__(blocks)
