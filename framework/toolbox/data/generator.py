from abc import ABC, abstractmethod

from framework.toolbox.data.dataset import Dataset


class DatasetGenerator(ABC):
    def __init__(self, name: str, path: str, batch_size: int, random_state: int):
        self.name = name
        self.path = path
        self.batch_size = batch_size
        self.seed = random_state
        self.datasets = []

    @abstractmethod
    def generate(self, train_proportion: float, validation_proportion: float, test_proportion: float) -> 'DatasetGenerator':
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
