from abc import ABC, abstractmethod


class Block(ABC):
    @abstractmethod
    def get_layers(self):
        pass
