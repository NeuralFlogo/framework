from abc import ABC, abstractmethod


class Device(ABC):
    def __init__(self, device: int):
        self.device = device

    @abstractmethod
    def get(self):
        pass
