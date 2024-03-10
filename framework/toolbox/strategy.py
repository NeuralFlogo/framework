from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset


class Strategy(ABC):
    @abstractmethod
    def evaluate(self, test_set: Dataset, model: Model):
        pass
