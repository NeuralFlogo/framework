from abc import ABC, abstractmethod
from typing import List


class Dataset(ABC):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def batches(self) -> List['Batch']:
        pass

    class Batch(ABC):
        @abstractmethod
        def inputs(self):
            pass

        @abstractmethod
        def targets(self):
            pass
