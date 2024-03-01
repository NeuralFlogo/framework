from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def compute(self, outputs, targets):
        pass
