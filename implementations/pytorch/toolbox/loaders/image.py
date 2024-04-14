import random
import string

from sklearn.model_selection import train_test_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.loader import DatasetLoader
from implementations.pytorch.toolbox.datasets.image import PytorchImageDataset

IMAGE_DELIMITER = "\t"
DATASET_FILENAME = "/dataset.txt"


class PytorchImageDatasetLoader(DatasetLoader):

    def __init__(self, path: string, batch_size: int, seed: int):
        super().__init__(path, batch_size, seed)
        self.__images = []
        self.__datasets = []
        self.__load()
        self.__num_classes = self.__get_num_classes()

    def load(self, train_proportion: float, validation_proportion: float, test_proportion: float):
        self.__shuffle()
        train_data, test_data = train_test_split(self.__images, test_size=test_proportion, random_state=self.seed)
        train_data, val_data = train_test_split(train_data, test_size=validation_proportion, random_state=self.seed)
        self.__datasets.append(self.__create_dataset(train_data))
        self.__datasets.append(self.__create_dataset(val_data))
        self.__datasets.append(self.__create_dataset(test_data))
        return self

    def __create_dataset(self, images):
        return PytorchImageDataset(self.batch_size, images)

    def train(self) -> 'Dataset':
        return self.__datasets[0]

    def validation(self) -> 'Dataset':
        return self.__datasets[1]

    def test(self) -> 'Dataset':
        return self.__datasets[2]

    def __load(self):
        with open(self.path + DATASET_FILENAME, "r") as file:
            header = True
            for line in file:
                if header:
                    header = False
                else:
                    self.__process_line(line)

    def __process_line(self, line):
        split_line = line.split(IMAGE_DELIMITER)
        self.__images.append((self.path + split_line[0], int(split_line[1])))

    def __shuffle(self):
        random.seed(self.seed)
        random.shuffle(self.__images)

    def __get_num_classes(self):
        return len(set([label[1] for label in self.__images]))
