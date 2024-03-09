from abc import ABC, abstractmethod


class ResultSaver(ABC):

    def __init__(self, path):
        self.laboratory_name = None
        self.eras = None
        self.path = path

    def set_laboratory_name(self, name):
        self.laboratory_name = name

    def set_era(self, eras):
        self.eras = eras

    @abstractmethod
    def save_epoch(self, architecture_name: str, experiment_name: str, epoch: int, train_measurement: float, valid_measurement: float):
        pass

    @abstractmethod
    def save_batch(self, architecture_name: str, experiment_name: str, epoch: int, batch: int, measurement: float):
        pass
