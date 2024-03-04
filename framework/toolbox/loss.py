from abc import ABC, abstractmethod
from typing import Union, List


class LossFunction(ABC):
    @abstractmethod
    def compute(self, predictions, targets) -> Union[float, List[float]]:
        pass
