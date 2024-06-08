from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.toolbox.device import Device


class Model(ABC):
    def __init__(self, architecture: Architecture, device: Device):
        self.architecture = architecture
        self.device = device

    @abstractmethod
    def weights(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass
