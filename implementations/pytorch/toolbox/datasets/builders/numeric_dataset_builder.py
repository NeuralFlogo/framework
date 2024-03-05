import string
import pandas as pd
from sklearn.model_selection import train_test_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.dataset_builder import DatasetBuilder
from implementations.pytorch.toolbox.datasets.datasets.numeric_dataset import PytorchNumericDataset

PREDICTION_COLUMN_NAME = "prediction"


class PytorchNumericDatasetBuilder(DatasetBuilder):

    def __init__(self, path: string, batch_size: int, seed: int):
        super().__init__(path, batch_size, seed)
        self.__pandas_dataset = self.__load()
        self.__datasets = []

    def __load(self):
        return pd.read_csv(self.path)

    def build(self, train_proportion: float, validation_proportion: float, test_proportion: float):
        train_data, test_data = train_test_split(self.__pandas_dataset, test_size=test_proportion,
                                                 random_state=self.seed)
        train_data, val_data = train_test_split(train_data, test_size=validation_proportion, random_state=self.seed)
        self.__datasets.append(self.__create_dataset(train_data))
        self.__datasets.append(self.__create_dataset(val_data))
        self.__datasets.append(self.__create_dataset(test_data))
        return self

    def __create_dataset(self, pandas_dataset):
        return PytorchNumericDataset(pandas_dataset, self.batch_size)

    def train(self) -> 'Dataset':
        return self.__datasets[0]

    def evaluation(self) -> 'Dataset':
        return self.__datasets[1]

    def test(self) -> 'Dataset':
        return self.__datasets[2]
