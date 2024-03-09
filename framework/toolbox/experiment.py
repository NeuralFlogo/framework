from abc import ABC, abstractmethod
from typing import Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset
from framework.toolbox.loss import LossFunction
from framework.toolbox.optimizer import Optimizer
from framework.toolbox.checkpoint_saver import CheckpointSaver
from framework.toolbox.result_saver import ResultSaver
from framework.toolbox.stopper import EarlyStopper


class Experiment(ABC):
    def __init__(self, name: str, optimizer: Optimizer, loss_function: LossFunction, stopper: EarlyStopper, checkpoint_saver: CheckpointSaver):
        self.name = name
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.stopper = stopper
        self.checkpoint_saver = checkpoint_saver

    @abstractmethod
    def run(self, epochs: int, training_set: Dataset, validation_set: Dataset, architecture: Architecture, result_saver: ResultSaver) -> Tuple[float, Model]:
        pass
