from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.toolbox.optimizer import Optimizer


class Saver(ABC):
    @abstractmethod
    def save(self, architecture: Architecture, optimizer: Optimizer = None):
        pass
