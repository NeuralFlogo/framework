import ast
import random

import pandas as pd

from framework.toolbox.data.loader import DatasetLoader
from implementations.pytorch.toolbox.data.datasets.numeric import PytorchNumericDataset

Delimiter = "delimiter"
Extension = ".csv"
TargetColumn = "prediction"
TargetColumnParameter = "target"
TypeColumn = "column"


class PytorchNumericDatasetLoader(DatasetLoader):
    def create_dataset(self, batch_size: int, data):
        return PytorchNumericDataset(batch_size, data, self.__target_column())

    def load(self):
        if self.__are_list():
            data = pd.read_csv(self.filename(Extension), delimiter=self.__delimiter(), header=0)
            for column in data.columns:
                if self.__is_list_string(data[column][0]):
                    data[column] = [ast.literal_eval(element) for element in data[column]]
        else:
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

    def __are_list(self):
        return self.metadata.get(TypeColumn, "numeric") == "list"

    def __is_list_string(self, value):
        try:
            result = eval(value)
            return isinstance(result, list)
        except:
            return False
