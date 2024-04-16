from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    def __init__(self, path: str, name: str, random_state: int, metadata):
        self.path = path
        self.name = name
        self.random_state = random_state
        self.metadata = metadata

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def create_dataset(self, batch_size: int, data):
        pass

    def filename(self, extension: str):
        return self.path + self.name + extension
