from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.optimizer import Optimizer


class ModelSaver(ABC):
    @abstractmethod
    def save(self, experiment: str, model: Model, optimizer: Optimizer):
        pass

    @abstractmethod
    def latest_checkpoint(self, experiment: str):
        pass
