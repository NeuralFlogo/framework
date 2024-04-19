from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.device import Device


class ModelLoader(ABC):
    @abstractmethod
    def load(self, path: str, architecture: Architecture, device: Device) -> Model:
        pass
