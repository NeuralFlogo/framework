from abc import ABC
from typing import List, Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.device import Device
from framework.toolbox.experiment import Experiment
from framework.toolbox.data.generator import DatasetGenerator
from framework.toolbox.loader import ModelLoader
from framework.toolbox.logger import Logger
from framework.toolbox.strategies.classification import ClassificationStrategy
from framework.toolbox.strategy import Strategy


class Laboratory(ABC):
    def __init__(self, name: str, eras: int, epochs: int, datagen: DatasetGenerator, architecture: Architecture, experiments: List[Experiment], strategy: Strategy, logger: Logger, loader: ModelLoader, device: Device):
        self.name = name
        self.eras = eras
        self.epochs = epochs
        self.datagen = datagen
        self.architecture = architecture
        self.experiments = experiments
        self.strategy = strategy
        self.logger = logger
        self.loader = loader
        self.device = device
        self.logger.set_laboratory_name(self.name)

    def explore(self):
        performances = []
        for era in range(1, self.eras + 1):
            self.logger.set_era(era)
            for experiment in self.experiments:
                performances.append(
                    experiment.run(epochs=self.epochs,
                                   training_set=self.datagen.train(),
                                   validation_set=self.datagen.validation(),
                                   architecture=self.architecture,
                                   logger=self.logger,
                                   loader=self.loader,
                                   device=self.device))
        self.logger.set_era(-1)
        self.__log_test_results(performances)

    def __log_test_results(self, performances: List[Tuple[Model, float]]):
        experiment = self.__best_experiment(performances)
        self.logger.log_test(architecture=self.architecture.name,
                             experiment=experiment.name,
                             strategy=self.__strategy_type(),
                             measurement=self.strategy.evaluate(self.datagen.test(), performances[self.experiments.index(experiment)][0], self.device))

    def __best_experiment(self, performances: List[Tuple[Model, float]]) -> Experiment:
        if isinstance(self.strategy, ClassificationStrategy):
            return self.experiments[performances.index(max(performances, key=lambda x: x[1]))]
        return self.experiments[performances.index(min(performances, key=lambda x: x[1]))]

    def __strategy_type(self) -> str:
        return self.strategy.__class__.__bases__[0].__name__.strip("Strategy").lower()
