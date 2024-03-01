from abc import ABC, abstractmethod


class EarlyStopper(ABC):
    def __init__(self, patience: int, delta: float):
        self.tolerance = patience
        self.delta = delta

    @abstractmethod
    def should_stop(self, loss: float) -> bool:
        pass
