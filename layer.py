from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def get(self):
        pass
