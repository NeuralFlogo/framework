from abc import ABC, abstractmethod
from typing import Union

from framework.architecture.block import Block
from framework.architecture.layer import Layer
from framework.architecture.section import Section


class Architecture(ABC):
    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def attach(self, component: Union[Section, Block, Layer]) -> 'Architecture':
        pass
