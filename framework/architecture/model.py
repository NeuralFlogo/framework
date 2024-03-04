from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture


class Model(ABC):
    def __init__(self, architecture: Architecture):
        self.architecture = architecture

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def predict(self):
        pass
