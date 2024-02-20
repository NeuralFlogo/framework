from typing import Union

from torch.nn import Module, Sequential

from framework.architecture import Architecture
from implementations.pytorch.block import PytorchBlock
from implementations.pytorch.layer import PytorchLayer
from implementations.pytorch.section import PytorchSection


class PytorchArchitecture(Module, Architecture):
    def __init__(self):
        super(PytorchArchitecture, self).__init__()
        self.components = Sequential()

    def attach(self, component: Union[PytorchSection, PytorchBlock, PytorchLayer]) -> 'PytorchArchitecture':
        self.components.append(component)
        return self

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x
