import torch

from framework.toolbox.experiment import Experiment
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchCheckpointSaver
from implementations.pytorch.toolbox.logger import PytorchLogger

BATCH_TO_SAVE = 10


class PytorchExperiment(Experiment):
    def __init__(self, name: str, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, stopper: EarlyStopper,
                 saver: PytorchCheckpointSaver):
        super().__init__(name, optimizer, loss_function, stopper, saver)

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset,
            architecture: PytorchArchitecture, logger: PytorchLogger):
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self.__train(epoch, training_set, architecture, logger)
            valid_loss = self.__validate(validation_set, architecture)
            logger.log_epoch(architecture.name, self.name, epoch, train_loss, valid_loss)
            if self.__is_checkpoint(valid_loss, best_loss):
                best_loss = valid_loss
                self.saver.save(PytorchModel(architecture), self.optimizer)
            if self.stopper.should_stop(valid_loss):
                self.saver.save(PytorchModel(architecture), self.optimizer)
                break
        return valid_loss, PytorchModel(architecture)

    def __train(self, epoch: int, dataset: PytorchDataset, architecture: PytorchArchitecture, logger: PytorchLogger):
        loss, previous_batch_loss = 0., 0.
        architecture.train(True)
        for i, batch in enumerate(dataset.batches()):
            outputs = architecture(batch.inputs())
            loss += self.loss_function.compute(outputs, batch.targets(), True)
            self.optimizer.move()
            if (i + 1) % BATCH_TO_SAVE == 0:
                self.__log_batch(epoch, i, architecture.name, self.__get_batch_loss(loss, previous_batch_loss), logger)
                previous_batch_loss = loss
        architecture.train(False)
        return loss / len(dataset.batches())

    def __get_batch_loss(self, loss, previous_batch_loss):
        return (loss - previous_batch_loss) / BATCH_TO_SAVE

    def __validate(self, dataset: PytorchDataset, architecture: PytorchArchitecture):
        loss = 0.
        architecture.eval()
        with torch.no_grad():
            for batch in dataset.batches():
                outputs = architecture(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        return loss / len(dataset.batches())

    def __is_checkpoint(self, loss, best_loss):
        return loss < best_loss

    def __log_batch(self, epoch: int, batch: int, architecture: str, loss: float, logger: PytorchLogger):
        logger.log_batch(architecture, self.name, epoch, batch + 1, loss)
