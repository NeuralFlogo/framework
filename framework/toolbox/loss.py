from abc import ABC, abstractmethod
from typing import Union, List


class LossFunction(ABC):
    @abstractmethod
    def compute(self, predictions, targets, training: bool) -> Union[float, List[float]]:
        pass
