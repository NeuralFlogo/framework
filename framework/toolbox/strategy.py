from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.data.dataset import Dataset
from framework.toolbox.device import Device


class Strategy(ABC):
    @abstractmethod
    def evaluate(self, test_set: Dataset, model: Model, device: Device) -> float:
        pass
