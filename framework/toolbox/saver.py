from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.optimizer import Optimizer


class CheckpointSaver(ABC):
    @abstractmethod
    def save(self, model: Model, optimizer: Optimizer):
        pass
