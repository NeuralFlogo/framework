from abc import ABC, abstractmethod
from typing import Tuple

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset
from framework.toolbox.loss import LossFunction
from framework.toolbox.optimizer import Optimizer
from framework.toolbox.saver import CheckpointSaver
from framework.toolbox.stopper import EarlyStopper


class Experiment(ABC):
    def __init__(self, loss_function: LossFunction, stopper: EarlyStopper,
                 checkpoint_saver: CheckpointSaver):
        self.loss_function = loss_function
        self.stopper = stopper
        self.saver = checkpoint_saver

    @abstractmethod
    def run(self, epochs: int, training_dataset: Dataset, eval_dataset: Dataset, architecture: Architecture) -> Tuple[Model, float]:
        pass

    @abstractmethod
    def test(self, dataset: Dataset, model: Model) -> float:
        pass
