from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.optimizer import Optimizer


class CheckpointSaver(ABC):
    @abstractmethod
    def save(self, model: Model, optimizer: Optimizer):
        pass
