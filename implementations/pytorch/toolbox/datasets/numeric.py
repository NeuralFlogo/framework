import numpy as np
import torch

from implementations.pytorch.toolbox.dataset import PytorchDataset

TargetColumn = "prediction"


class PytorchNumericDataset(PytorchDataset):
    def __init__(self, batch_size: int, data):
        super().__init__(batch_size)
        self.__data = data

    def __len__(self):
        return self.__data.shape[0]

    def __getitem__(self, idx):
        row = self.__data.iloc[idx]
        prediction = row[TargetColumn]
        data = row.drop(TargetColumn)
        return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(prediction, dtype=torch.float32)
