from abc import ABC, abstractmethod
from typing import List, Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset
from framework.toolbox.experiment import Experiment


class Laboratory(ABC):
    def __init__(self, epochs: int, dataset: Dataset, architecture: Architecture, experiments: List[Experiment]):
        self.epochs = epochs
        self.dataset = dataset
        self.architecture = architecture
        self.experiments = experiments

    def explore(self):
        performances = []
        for experiment in self.experiments:
            performances.append(experiment.run(self.epochs, self.dataset.train(), self.architecture))
        experiment, model = self.__best_experiment(performances)
        return experiment.test(self.dataset, model)

    def __best_experiment(self, performances: List[Tuple[Model, float]]) -> Tuple[Experiment, Model]:
        return self.experiments[performances.index(min(performances, key=lambda x: x[1]))], min(performances, key=lambda x: x[1])[0]
