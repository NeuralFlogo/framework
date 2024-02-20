from abc import ABC, abstractmethod
from typing import Union

from framework.block import Block
from framework.layer import Layer
from framework.section import Section


class Architecture(ABC):
    def __init__(self, name: str = None):
        self.name = name

    @abstractmethod
    def attach(self, component: Union[Section, Block, Layer]) -> 'Architecture':
        pass
