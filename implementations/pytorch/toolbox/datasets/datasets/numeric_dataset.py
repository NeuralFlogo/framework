from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

from framework.toolbox.dataset import Dataset
from implementations.pytorch.toolbox.datasets.batch import PytorchBatch

PREDICTION_COLUMN_NAME = "prediction"


class PytorchNumericDataset(torch.utils.data.Dataset, Dataset):

    def __init__(self, pandas_dataset, batch_size):
        super().__init__(batch_size)
        self.__pandas_dataset = pandas_dataset

    def __len__(self):
        return self.__pandas_dataset.shape[0]

    def __getitem__(self, idx):
        row = self.__pandas_dataset.iloc[idx]
        prediction = row[PREDICTION_COLUMN_NAME]
        data = row.drop(columns=[PREDICTION_COLUMN_NAME])
        return torch.tensor(data), torch.tensor(prediction)

    def batches(self) -> List[PytorchBatch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchBatch(inputs, targets))
        return batches
