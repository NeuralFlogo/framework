import numpy as np
import pandas as pd
import torch

from implementations.pytorch.toolbox.data.dataset import PytorchDataset


class PytorchNumericDataset(PytorchDataset):
    def __init__(self, batch_size: int, data, target_column):
        super().__init__(batch_size)
        self.__data = data
        self.__target_column = target_column

    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, idx):
        row = self.__data.iloc[idx]
        prediction = row[self.__target_column]
        data = row.drop(self.__target_column)
        if isinstance(data, pd.Series):
            return torch.tensor([row for i, row in data.items()], dtype=torch.float), torch.tensor(prediction, dtype=torch.float) if type(
            prediction.item()) == float else torch.tensor(prediction, dtype=torch.long)
        return torch.tensor(np.array(data), dtype=torch.float), torch.tensor(prediction, dtype=torch.float) if type(
            prediction.item()) == float else torch.tensor(prediction, dtype=torch.long)
