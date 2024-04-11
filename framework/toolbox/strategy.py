from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset
from framework.toolbox.logger import Logger


class Strategy(ABC):
    @abstractmethod
    def evaluate(self, test_set: Dataset, model: Model, architecture_name: str, experiment_name: str, logger: Logger):
        pass
