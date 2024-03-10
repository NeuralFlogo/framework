import string
from abc import ABC, abstractmethod

from framework.toolbox.dataset import Dataset


class DatasetLoader(ABC):
    def __init__(self, path: string, batch_size: int, seed: int):
        self.path = path
        self.batch_size = batch_size
        self.seed = seed

    @abstractmethod
    def load(self, train_proportion: float, validation_proportion: float, test_proportion: float) -> 'DatasetLoader':
        pass

    @abstractmethod
    def train(self) -> 'Dataset':
        pass

    @abstractmethod
    def validation(self) -> 'Dataset':
        pass

    @abstractmethod
    def test(self) -> 'Dataset':
        pass
