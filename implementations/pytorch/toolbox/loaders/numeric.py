import random

import pandas as pd

from framework.toolbox.loader import DatasetLoader
from implementations.pytorch.toolbox.datasets.numeric import PytorchNumericDataset

DELIMITER_PARAMETER_NAME = "delimiter"
HEADER_PARAMETER_NAME = "header"
FILE_EXTENSION = ".csv"


class PytorchNumericDatasetLoader(DatasetLoader):
    def create_dataset(self, batch_size, dataset):
        return PytorchNumericDataset(batch_size, dataset)

    def load(self):
        return self.__shuffle(pd.read_csv(self.filename(FILE_EXTENSION), delimiter=self.__delimiter(), header=self.__header()))

    def __header(self):
        return 0 if bool(self.metadata[HEADER_PARAMETER_NAME]) else None

    def __delimiter(self):
        return self.metadata[DELIMITER_PARAMETER_NAME]

    def __shuffle(self, dataset):
        random.seed(self.seed)
        return dataset.sample(frac=1).reset_index(drop=True)
