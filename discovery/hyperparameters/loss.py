from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(self, predictions, labels):
        pass
