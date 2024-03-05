from typing import List

import torch
from torch.utils.data import DataLoader

from framework.toolbox.batch import Batch
from framework.toolbox.dataset import Dataset
from implementations.pytorch.toolbox.datasets.batch import PytorchBatch


class PytorchImageDataset(torch.utils.data.Dataset, Dataset):

    def __init__(self, batch_size, images):
        super().__init__(batch_size)
        self.__images = images

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def batches(self) -> List[Batch]:
        batches = []
        for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True):
            batches.append(PytorchBatch(inputs, targets))
        return batches