import string
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.dataset_builder import DatasetBuilder
from implementations.pytorch.toolbox.datasets.datasets.numeric_dataset import PytorchNumericDataset

PREDICTION_COLUMN_NAME = "prediction"


class PytorchNumericDatasetBuilder(DatasetBuilder):

    def __init__(self, path: string, batch_size: int, seed: int):
        super().__init__(path, batch_size, seed)
        self.__pandas_dataset = self.__load()
        self.__datasets = None

    def __load(self):
        return pd.read_csv(self.path)

    def build(self, train_proportion: float, validation_proportion: float, test_proportion: float):
        dataset = self.__create_dataset()
        self.__datasets = random_split(dataset, [train_proportion, validation_proportion, test_proportion])
        return self

    def __create_dataset(self):
        return PytorchNumericDataset(self.batch_size, self.__pandas_dataset)

    def train(self) -> 'Dataset':
        return self.__datasets[0]

    def evaluation(self) -> 'Dataset':
        return self.__datasets[1]

    def test(self) -> 'Dataset':
        return self.__datasets[2]
