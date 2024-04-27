import random

import pandas as pd

from framework.toolbox.data.loader import DatasetLoader
from implementations.pytorch.toolbox.data.datasets.numeric import PytorchNumericDataset

Delimiter = "delimiter"
Header = "header"
Extension = ".csv"
TargetColumn = "prediction"
TargetColumnParameter = "target"


class PytorchNumericDatasetLoader(DatasetLoader):
    def create_dataset(self, batch_size: int, data):
        return PytorchNumericDataset(batch_size, data, self.__target_column())

    def load(self):
        data = pd.read_csv(self.filename(Extension), delimiter=self.__delimiter(), header=0)
        if self.metadata['task'] == 'classification':
            data[self.__target_column()] = data[self.__target_column()].astype(int)
        return self.__shuffle(data)

    def __delimiter(self):
        return self.metadata.get(Delimiter, "\t")

    def __target_column(self):
        return self.metadata.get(TargetColumnParameter, TargetColumn)

    def __shuffle(self, dataset):
        random.seed(self.random_state)
        return dataset.sample(frac=1).reset_index(drop=True)
