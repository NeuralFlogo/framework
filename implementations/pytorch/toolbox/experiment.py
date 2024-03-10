import torch

from framework.toolbox.experiment import Experiment
from framework.toolbox.logger import Logger
from framework.toolbox.stopper import EarlyStopper
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchModelSaver

BATCH_TO_SAVE = 10  # TODO make it an argument


class PytorchExperiment(Experiment):
    def __init__(self, name: str, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction,
                 stopper: EarlyStopper, saver: PytorchModelSaver):
        super().__init__(name, optimizer, loss_function, stopper, saver)
        self.best_loss = float("inf")

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset,
            architecture: PytorchArchitecture, logger: Logger):
        for epoch in range(1, epochs + 1):
            train_loss = self.__train(epoch, training_set, architecture, logger)
            valid_loss = self.__validate(validation_set, architecture)
            logger.log_epoch(architecture.name, self.name, epoch, train_loss, valid_loss)
            if self.__is_checkpoint(valid_loss):
                self.best_loss = valid_loss
                self.saver.save(PytorchModel(architecture), self.optimizer)
            if self.stopper.should_stop(valid_loss):
                self.saver.save(PytorchModel(architecture), self.optimizer)
                break
        return valid_loss, PytorchModel(architecture)

    def __train(self, epoch: int, dataset: PytorchDataset, architecture: PytorchArchitecture, logger: Logger):
        running_loss, last_loss = 0., 0.
        architecture.train(True)
        for i, batch in enumerate(dataset.batches(), start=1):
            outputs = architecture(batch.inputs())
            running_loss += self.loss_function.compute(outputs, batch.targets(), True)
            self.optimizer.move()
            if i % BATCH_TO_SAVE == 0:
                logger.log_batch(epoch, i, architecture.name, self.__get_batch_loss(running_loss, last_loss))
                last_loss = running_loss
        architecture.train(False)
        return running_loss / len(dataset.batches())

    def __validate(self, dataset: PytorchDataset, architecture: PytorchArchitecture):
        loss = 0.
        architecture.eval()
        with torch.no_grad():
            for batch in dataset.batches():
                outputs = architecture(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        return loss / len(dataset.batches())

    def __is_checkpoint(self, loss: float):
        return loss < self.best_loss

    def __get_batch_loss(self, running_loss: float, last_loss: float):
        return (running_loss - last_loss) / BATCH_TO_SAVE
