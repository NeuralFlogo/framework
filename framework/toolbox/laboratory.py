from abc import ABC
from typing import List, Dict, Tuple

from framework.toolbox.data.generator import DatasetGenerator
from framework.toolbox.device import Device
from framework.toolbox.experiment import Experiment
from framework.toolbox.loader import ModelLoader
from framework.toolbox.logger import Logger
from framework.toolbox.strategy import Strategy


class Laboratory(ABC):
    def __init__(self, name: str, eras: int, epochs: int, datagen: DatasetGenerator, experiments: List[Experiment], strategy: Strategy, logger: Logger, loader: ModelLoader, device: Device):
        self.name = name
        self.eras = eras
        self.epochs = epochs
        self.datagen = datagen
        self.experiments = experiments
        self.strategy = strategy
        self.logger = logger
        self.loader = loader
        self.device = device
        self.logger.set_laboratory_name(self.name)

    def explore(self):
        performances = {}
        for era in range(1, self.eras + 1):
            self.logger.set_era(era)
            for experiment in self.experiments:
                model = experiment.run(epochs=self.epochs,
                                       training_set=self.datagen.train(),
                                       validation_set=self.datagen.validation(),
                                       logger=self.logger,
                                       loader=self.loader,
                                       device=self.device)
                performances[experiment] = self.strategy.evaluate(self.datagen.test(), model, self.device)
        self.logger.set_era(-1)
        self.__log_test_results(performances)

    def __log_test_results(self, performances: Dict[Experiment, float]):
        experiment, test = self.__best_experiment(performances)
        self.logger.log_test(experiment=experiment.name,
                             strategy=self.__strategy_type(),
                             measurement=test)

    def __best_experiment(self, performances: Dict[Experiment, float]) -> Tuple[Experiment, float]:
        experiment = max(performances, key=performances.get)
        return experiment, performances[experiment]

    def __strategy_type(self) -> str:
        return self.strategy.__class__.__bases__[0].__name__.strip("Strategy").lower()
