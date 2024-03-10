from abc import ABC

from framework.toolbox.dataset import Dataset


class PytorchDataset(Dataset, ABC):
    class PytorchBatch(Dataset.Batch):
        def __init__(self, inputs, targets):
            self.__inputs = inputs
            self.__targets = targets

        def inputs(self):
            return self.__inputs

        def targets(self):
            return self.__targets
