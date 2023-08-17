from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def build(self, layers):
        pass
