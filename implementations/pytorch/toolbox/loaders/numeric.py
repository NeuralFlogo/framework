import string
import pandas as pd
from sklearn.model_selection import train_test_split

from framework.toolbox.dataset import Dataset
from framework.toolbox.loader import DatasetLoader
from implementations.pytorch.toolbox.datasets.datasets.numeric import PytorchNumericDataset

PREDICTION_COLUMN_NAME = "prediction"


class PytorchNumericDatasetLoader(DatasetLoader):

    def __init__(self, path: string, batch_size: int, seed: int):
        super().__init__(path, batch_size, seed)
        self.__pandas_dataset = self.__load()
        self.__datasets = []

    def __load(self):
        return pd.read_csv(self.path)

    def load(self, train_proportion: float, validation_proportion: float, test_proportion: float):
        train_data, test_data = train_test_split(self.__pandas_dataset, test_size=test_proportion,
                                                 random_state=self.seed)
        train_data, val_data = train_test_split(train_data, test_size=validation_proportion, random_state=self.seed)
        self.__datasets.append(self.__create_dataset(train_data))
        self.__datasets.append(self.__create_dataset(val_data))
        self.__datasets.append(self.__create_dataset(test_data))
        return self

    def __create_dataset(self, pandas_dataset):
        return PytorchNumericDataset(self.batch_size, pandas_dataset)

    def train(self) -> 'Dataset':
        return self.__datasets[0]

    def validation(self) -> 'Dataset':
        return self.__datasets[1]

    def test(self) -> 'Dataset':
        return self.__datasets[2]
