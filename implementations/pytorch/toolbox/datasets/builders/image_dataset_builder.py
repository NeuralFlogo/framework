import string

from torch.utils.data import random_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.dataset_builder import DatasetBuilder
from implementations.pytorch.toolbox.datasets.datasets.image_dataset import PytorchImageDataset

IMAGE_DELIMITER = " "


class PytorchImageDatasetBuilder(DatasetBuilder):

    def __init__(self, path: string, batch_size: int, seed: int):
        super().__init__(path, batch_size, seed)
        self.__images = []
        self.__load()
        self.__datasets = None

    def build(self, train_proportion: float, validation_proportion: float, test_proportion: float):
        dataset = self.__create_dataset()
        self.__datasets = random_split(dataset, [train_proportion, validation_proportion, test_proportion])
        return self

    def __create_dataset(self):
        return PytorchImageDataset(self.batch_size, self.__images)

    def train(self) -> 'Dataset':
        return self.__datasets[0]

    def evaluation(self) -> 'Dataset':
        return self.__datasets[1]

    def test(self) -> 'Dataset':
        return self.__datasets[2]

    def __load(self):
        with open(self.path, "r") as file:
            for line in file:
                self.__process_line(line)

    def __process_line(self, line):
        split_line = line.split(IMAGE_DELIMITER)
        self.__images.append((split_line[0], split_line[1]))
