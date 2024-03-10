from abc import ABC
from typing import List, Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.loader import DatasetLoader
from framework.toolbox.experiment import Experiment
from framework.toolbox.logger import Logger
from framework.toolbox.strategy import Strategy


class Laboratory(ABC):
    def __init__(self, name: str, epochs: int, dataset: DatasetLoader, architecture: Architecture, experiments: List[Experiment],
                 strategy: Strategy, logger: Logger, eras: int = 1):
        self.name = name
        self.eras = eras
        self.epochs = epochs
        self.dataset = dataset
        self.architecture = architecture
        self.experiments = experiments
        self.strategy = strategy
        self.logger = logger

    def explore(self):
        self.logger.set_laboratory_name(self.name)
        performances = []
        for era in range(1, self.eras + 1):
            self.logger.set_era(era)
            for experiment in self.experiments:
                performances.append(experiment.run(self.epochs,
                                                   self.dataset.train(),
                                                   self.dataset.validation(),
                                                   self.architecture,
                                                   self.logger))
        return self.strategy.evaluate(self.dataset.test(),
                                      self.__best_model(performances)) #TODO Log the test results

    def __best_model(self, performances: List[Tuple[float, Model]]) -> Model:
        return performances[performances.index(min(performances, key=lambda x: x[0]))][1]
