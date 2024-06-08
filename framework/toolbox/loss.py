from abc import ABC, abstractmethod
from typing import Union, List


class LossFunction(ABC):
    @abstractmethod
    def training_compute(self, predictions, targets) -> Union[float, List[float]]:
        pass

    @abstractmethod
    def validation_compute(self, predictions, targets) -> Union[float, List[float]]:
        pass
