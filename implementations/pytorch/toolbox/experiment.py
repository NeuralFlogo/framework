from typing import Tuple

import torch

from framework.toolbox.experiment import Experiment
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.data.dataset import PytorchDataset
from implementations.pytorch.toolbox.device import PytorchDevice
from implementations.pytorch.toolbox.loader import PytorchModelLoader
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver

SavePoint = 10


class PytorchExperiment(Experiment):
    def __init__(self, name: str, architecture: PytorchArchitecture, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, saver: PytorchModelSaver, stopper: EarlyStopper = None):
        super().__init__(name=name, architecture=architecture, optimizer=optimizer, loss_function=loss_function, stopper=stopper, saver=saver)
        self.best_loss = float("inf")

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset, logger: Logger, loader: PytorchModelLoader, device: PytorchDevice) -> Tuple[PytorchModel, float]:
        validation_loss = float("inf")
        self.architecture.to(device.get())
        for epoch in range(1, epochs + 1):
            training_loss = self.__train(epoch, training_set, logger, device)
            validation_loss = self.__validate(validation_set, device)
            logger.log_epoch(self.architecture.name, self.name, epoch, training_loss, validation_loss)
            if self.__is_checkpoint(validation_loss):
                self.best_loss = validation_loss
                self.saver.save(self.name, PytorchModel(architecture=self.architecture, device=device), self.optimizer)
            if self.stopper and self.stopper.should_stop(validation_loss):
                break
        return self.__get_best_model(loader=loader, device=device), validation_loss

    def __train(self, epoch: int, dataset: PytorchDataset, logger: Logger, device: PytorchDevice):
        running_loss, last_loss = 0., 0.
        self.architecture.train(True)
        for i, batch in enumerate(dataset.batches(), start=1):
            outputs = self.architecture(batch.inputs().to(device.get()))
            running_loss += self.loss_function.training_compute(outputs, batch.targets().to(device.get()))
            self.optimizer.move()
            if i % SavePoint == 0:
                logger.log_batch(self.architecture.name, self.name, epoch, i, self.__get_batch_loss(running_loss, last_loss))
                last_loss = running_loss
        self.architecture.train(False)
        return running_loss / len(dataset.batches())

    def __validate(self, dataset: PytorchDataset, device: PytorchDevice):
        loss = 0.
        self.architecture.eval()
        with torch.no_grad():
            for batch in dataset.batches():
                outputs = self.architecture(batch.inputs().to(device.get()))
                loss += self.loss_function.validation_compute(outputs, batch.targets().to(device.get()))
        return loss / len(dataset.batches())

    def __is_checkpoint(self, loss: float):
        return loss < self.best_loss

    def __get_batch_loss(self, running_loss: float, last_loss: float):
        return (running_loss - last_loss) / SavePoint

    def __get_best_model(self, loader: PytorchModelLoader, device: PytorchDevice) -> PytorchModel:
        return loader.load(path=self.saver.latest_checkpoint(self.name), architecture=self.architecture, device=device)
