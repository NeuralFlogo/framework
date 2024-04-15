from abc import ABC, abstractmethod


class DatasetLoader(ABC):

    def __init__(self, path, name, seed, metadata):
        self.path = path
        self.name = name
        self.seed = seed
        self.metadata = metadata

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def create_dataset(self, batch_size, data):
        pass

    def filename(self, file_extension):
        return self.path + self.name + file_extension
