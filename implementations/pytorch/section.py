from typing import List

from torch import nn


from framework.section import Section
from implementations.pytorch.block import PytorchBlock


class PytorchSection(Section, nn.Module):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchSection, self).__init__(blocks)
