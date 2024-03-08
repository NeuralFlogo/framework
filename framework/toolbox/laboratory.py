from abc import ABC
from typing import List, Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.dataset_builder import DatasetLoader
from framework.toolbox.experiment import Experiment
from framework.toolbox.strategy import Strategy


class Laboratory(ABC):
    def __init__(self, epochs: int, dataset: DatasetLoader, architecture: Architecture, experiments: List[Experiment],
                 strategy: Strategy, eras: int = 1):
        self.eras = eras
        self.epochs = epochs
        self.dataset = dataset
        self.architecture = architecture
        self.experiments = experiments
        self.strategy = strategy

    def explore(self):
        performances = []
        for _ in range(self.eras):
            for experiment in self.experiments:
                performances.append(experiment.run(self.epochs, self.dataset.train(), self.dataset.validation(), self.architecture))
        return self.strategy.evaluate(self.dataset.test(), self.__best_model(performances))

    def __best_model(self, performances: List[Tuple[float, Model]]) -> Model:
        return performances[performances.index(min(performances, key=lambda x: x[0]))][1]
