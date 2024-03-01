from typing import List

from torch import nn


from framework.architecture.section import Section
from implementations.pytorch.architecture.block import PytorchBlock


class PytorchSection(Section, nn.Module):
    def __init__(self, blocks: List[PytorchBlock]):
        super(PytorchSection, self).__init__(blocks)
