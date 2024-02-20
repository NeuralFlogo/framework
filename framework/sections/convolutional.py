from typing import List

from framework.blocks.convolutional import ConvolutionalBlock
from framework.section import Section


class ConvolutionalSection(Section):
    def __init__(self, blocks: List[ConvolutionalBlock]):
        super().__init__(blocks)
