from typing import List

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from framework.toolbox.data.dataset import Dataset


class PytorchDataset(Dataset, TorchDataset):
    def batches(self) -> List['PytorchBatch']:
        return [PytorchDataset.PytorchBatch(inputs, targets) for inputs, targets in DataLoader(self, batch_size=self.batch_size, shuffle=True)]

    class PytorchBatch(Dataset.Batch):
        def __init__(self, inputs, targets):
            self.__inputs = inputs
            self.__targets = targets

        def inputs(self):
            return self.__inputs

        def targets(self):
            return self.__targets
