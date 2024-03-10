from abc import ABC, abstractmethod
from typing import List


class Dataset(ABC):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @abstractmethod
    def batches(self) -> List['Batch']:
        pass

    class Batch(ABC):
        @abstractmethod
        def targets(self):
            pass

        @abstractmethod
        def inputs(self):
            pass
