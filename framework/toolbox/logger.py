from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self, path):
        self.laboratory = None
        self.era = None
        self.path = path

    def set_laboratory_name(self, name: str):
        self.laboratory = name

    def set_era(self, era: int):
        self.era = era

    @abstractmethod
    def log_epoch(self, architecture: str, experiment: str, epoch: int, train_measurement: float, valid_measurement: float):
        pass

    @abstractmethod
    def log_batch(self, architecture: str, experiment: str, epoch: int, batch: int, measurement: float):
        pass
