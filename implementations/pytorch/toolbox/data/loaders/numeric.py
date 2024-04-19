import random

import pandas as pd

from framework.toolbox.data.loader import DatasetLoader
from implementations.pytorch.toolbox.data.datasets.numeric import PytorchNumericDataset

Delimiter = "delimiter"
Header = "header"
Extension = ".csv"
TargetColumn = "prediction"


class PytorchNumericDatasetLoader(DatasetLoader):
    def create_dataset(self, batch_size: int, data):
        return PytorchNumericDataset(batch_size, data)

    def load(self):
        data = pd.read_csv(self.filename(Extension), delimiter=self.__delimiter(), header=self.__header())
        if self.metadata['task'] == 'classification':
            data[TargetColumn] = data[TargetColumn].astype(int)
        return self.__shuffle(data)

    def __header(self):
        return 0 if bool(self.metadata[Header]) else None

    def __delimiter(self):
        return self.metadata[Delimiter]

    def __shuffle(self, dataset):
        random.seed(self.random_state)
        return dataset.sample(frac=1).reset_index(drop=True)
