from abc import ABC, abstractmethod


class Section(ABC):
    @abstractmethod
    def get_layers(self):
        pass

    def __call__(self, x):
        return x.add(self)
