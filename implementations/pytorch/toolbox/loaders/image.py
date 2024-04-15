import random

from framework.toolbox.loader import DatasetLoader
from implementations.pytorch.toolbox.datasets.image import PytorchImageDataset

IMAGE_DELIMITER = "\t"
FILE_EXTENSION = ".txt"


class PytorchImageDatasetLoader(DatasetLoader):

    def __init__(self, path, name, seed, metadata):
        super().__init__(path, name, seed, metadata)
        self.__images = []

    def __shuffle(self):
        random.seed(self.seed)
        random.shuffle(self.__images)
        return self.__images

    def create_dataset(self, batch_size, images):
        return PytorchImageDataset(batch_size, images)

    def load(self):
        with open(self.filename(FILE_EXTENSION), "r") as file:
            for line in file:
                self.__process_line(line)
        return self.__shuffle()

    def __process_line(self, line):
        split_line = line.split(IMAGE_DELIMITER)
        self.__images.append((self.path + split_line[0], int(split_line[1])))
