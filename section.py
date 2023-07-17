from abc import ABC, abstractmethod


class Section(ABC):
    @abstractmethod
    def layers(self):
        pass

    def __call__(self, x):
        return x.add(self)
