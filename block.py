from abc import ABC, abstractmethod


class Block(ABC):
    @abstractmethod
    def layers(self):
        pass
