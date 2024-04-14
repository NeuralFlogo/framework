from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from implementations.pytorch.toolbox.dataset import PytorchDataset

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
        data = row.drop(PREDICTION_COLUMN_NAME)
        return torch.tensor(np.array(data), dtype=torch.float32), self.__create_prediction(prediction)

    def __create_prediction(self, prediction):
        return torch.tensor(prediction, dtype=torch.float32) if type(prediction) == float else torch.tensor(prediction, dtype=torch.long) #TODO Float for regression, Long for Classification

    def batches(self) -> List[PytorchDataset.PytorchBatch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchDataset.PytorchBatch(inputs, targets))
        return batches
