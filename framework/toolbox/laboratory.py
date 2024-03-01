from typing import List

from framework.architecture.architecture import Architecture
from framework.toolbox.dataset import Dataset
from framework.toolbox.experiment import Experiment


class Laboratory:
    def __init__(self, epochs: int, dataset: Dataset, architecture: Architecture, experiments: List[Experiment]):
        self.epochs = epochs
        self.dataset = dataset
        self.architecture = architecture
        self.experiments = experiments

    def explore(self):
        performances = []
        for experiment in self.experiments:
            performances.append(experiment.run(self.epochs, self.dataset, self.architecture))
        experiment = self.experiments[performances.index(max(performances))]
        self.__test(experiment)

    def __test(self, experiment: Experiment):
        pass
