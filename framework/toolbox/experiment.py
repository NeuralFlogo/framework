from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.toolbox.dataset import Dataset
from framework.toolbox.loss import LossFunction
from framework.toolbox.optimizer import Optimizer
from framework.toolbox.saver import Saver
from framework.toolbox.stopper import EarlyStopper


class Experiment(ABC):
    def __init__(self, optimizer: Optimizer, loss_function: LossFunction, stopper: EarlyStopper, saver: Saver):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.stopper = stopper
        self.saver = saver

    @abstractmethod
    def run(self, epochs: int, dataset: Dataset, architecture: Architecture) -> float:
        pass
