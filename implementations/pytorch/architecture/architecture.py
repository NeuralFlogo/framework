from typing import Union

from torch.nn import Module, Sequential

from framework.architecture.architecture import Architecture
from implementations.pytorch.architecture.block import PytorchBlock
from implementations.pytorch.architecture.layer import PytorchLayer
from implementations.pytorch.architecture.section import PytorchSection


class PytorchArchitecture(Module, Architecture):
    def __init__(self, name):
        Module.__init__(self)
        Architecture.__init__(self, name)
        self.components = Sequential()

    def attach(self, component: Union[PytorchSection, PytorchBlock, PytorchLayer]) -> 'PytorchArchitecture':
        self.components.append(component)
        return self

    def forward(self, x):
        return self.components(x)
