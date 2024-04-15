from abc import ABC, abstractmethod

from framework.architecture.model import Model
from framework.toolbox.experiment import Experiment


class ModelSaver(ABC):
    @abstractmethod
    def save(self, model: Model, experiment: Experiment):
        pass
