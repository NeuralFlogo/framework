from typing import Tuple

import torch

from framework.toolbox.experiment import Experiment
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.device import PytorchDevice
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver

SavePoint = 10


class PytorchExperiment(Experiment):
    def __init__(self, name: str, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, stopper: EarlyStopper, saver: PytorchModelSaver):
        super().__init__(name, optimizer, loss_function, stopper, saver)
        self.best_loss = float("inf")

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset, architecture: PytorchArchitecture, logger: Logger, device: PytorchDevice) -> Tuple[PytorchModel, float]:
        validation_loss = float("inf")
        architecture.to(device.get())
        for epoch in range(1, epochs + 1):
            training_loss = self.__train(epoch, training_set, architecture, logger, device)
            validation_loss = self.__validate(validation_set, architecture, device)
            logger.log_epoch(architecture.name, self.name, epoch, training_loss, validation_loss)
            if self.__is_checkpoint(validation_loss):
                self.best_loss = validation_loss
                self.saver.save(self.name, PytorchModel(architecture), self.optimizer)
            if self.stopper.should_stop(validation_loss):
                break
        return PytorchModel(architecture), validation_loss

    def __train(self, epoch: int, dataset: PytorchDataset, architecture: PytorchArchitecture, logger: Logger, device: PytorchDevice):
        running_loss, last_loss = 0., 0.
        architecture.train(True)
        for i, batch in enumerate(dataset.batches(), start=1):
            outputs = architecture(batch.inputs().to(device.get()))
            running_loss += self.loss_function.training_compute(outputs, batch.targets().to(device.get()))
            self.optimizer.move()
            if i % SavePoint == 0:
                logger.log_batch(architecture.name, self.name, epoch, i, self.__get_batch_loss(running_loss, last_loss))
                last_loss = running_loss
        architecture.train(False)
        return running_loss / len(dataset.batches())

    def __validate(self, dataset: PytorchDataset, architecture: PytorchArchitecture, device: PytorchDevice):
        loss = 0.
        architecture.eval()
        with torch.no_grad():
            for batch in dataset.batches():
                outputs = architecture(batch.inputs().to(device.get()))
                loss += self.loss_function.validation_compute(outputs, batch.targets().to(device.get()))
        return loss / len(dataset.batches())

    def __is_checkpoint(self, loss: float):
        return loss < self.best_loss

    def __get_batch_loss(self, running_loss: float, last_loss: float):
        return (running_loss - last_loss) / SavePoint
