from abc import ABC, abstractmethod

from framework.architecture.architecture import Architecture
from framework.architecture.model import Model
from framework.toolbox.data.dataset import Dataset
from framework.toolbox.device import Device
from framework.toolbox.loader import ModelLoader
from framework.toolbox.logger import Logger
from framework.toolbox.loss import LossFunction
from framework.toolbox.optimizer import Optimizer
from framework.toolbox.saver import ModelSaver
from framework.toolbox.stopper import EarlyStopper


class Experiment(ABC):
    def __init__(self, name: str, architecture: Architecture, optimizer: Optimizer, loss_function: LossFunction, stopper: EarlyStopper, saver: ModelSaver):
        self.name = name
        self.architecture = architecture
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.stopper = stopper
        self.saver = saver

    @abstractmethod
    def run(self, epochs: int, training_set: Dataset, validation_set: Dataset, logger: Logger, loader: ModelLoader, device: Device) -> Model:
        pass
