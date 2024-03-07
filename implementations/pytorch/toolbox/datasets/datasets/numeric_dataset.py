from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from implementations.pytorch.toolbox.datasets.batch import PytorchBatch
from implementations.pytorch.toolbox.datasets.datasets.pytorch_dataset import PytorchDataset

PREDICTION_COLUMN_NAME = "prediction"


class PytorchNumericDataset(PytorchDataset, Dataset):

    def __init__(self, batch_size, pandas_dataset):
        super().__init__(batch_size)
        self.__pandas_dataset = pandas_dataset

    def __len__(self):
        return self.__pandas_dataset.shape[0]

    def __getitem__(self, idx):
        row = self.__pandas_dataset.iloc[idx]
        prediction = row[PREDICTION_COLUMN_NAME]
        data = row.drop(columns=[PREDICTION_COLUMN_NAME])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(prediction, dtype=torch.float32)

    def batches(self) -> List[PytorchBatch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchBatch(inputs, targets))
        return batches
