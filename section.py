from abc import ABC, abstractmethod


class Section(ABC):
    @abstractmethod
    def layers(self):
        pass
