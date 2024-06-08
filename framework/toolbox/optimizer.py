from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def weights(self):
        pass
