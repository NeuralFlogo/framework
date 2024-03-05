from abc import ABC, abstractmethod
from typing import List

from framework.toolbox.batch import Batch


class Dataset(ABC):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    @abstractmethod
    def batches(self) -> List[Batch]:
        pass
