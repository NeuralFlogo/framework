import random

from framework.toolbox.data.loader import DatasetLoader
from implementations.pytorch.toolbox.data.datasets.image import PytorchImageDataset

Delimiter = "\t"
Extension = ".tsv"


class PytorchImageDatasetLoader(DatasetLoader):
    def __init__(self, path: str, name: str, random_state: int, metadata):
        super().__init__(path, name, random_state, metadata)
        self.__images = []

    def create_dataset(self, batch_size: int, images):
        return PytorchImageDataset(batch_size, images)

    def load(self):
        with open(self.filename(Extension), "r") as file:
            for line in file:
                self.__process_line(line)
        return self.__shuffle()

    def __process_line(self, line: str):
        split_line = line.split(Delimiter)
        self.__images.append((self.path + split_line[0], int(split_line[1])))

    def __shuffle(self):
        random.seed(self.random_state)
        random.shuffle(self.__images)
        return self.__images
