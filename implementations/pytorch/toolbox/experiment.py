import torch

from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset

from framework.toolbox.experiment import Experiment
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchSaver
from implementations.pytorch.toolbox.stopper import PytorchEarlyStopper


class PytorchExperiment(Experiment):
    def __init__(self, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, stopper: PytorchEarlyStopper,
                 checkpoint_saver: PytorchSaver):
        super().__init__(optimizer, loss_function, stopper, checkpoint_saver)

    def run(self, epochs: int, dataset: PytorchDataset, architecture: PytorchArchitecture):
        best_loss = float("inf")
        for epoch in range(epochs):
            print("-" * 25, 'Epoch {}'.format(epoch + 1), "-" * 29)
            train_loss = self.__train(dataset, architecture)
            valid_loss = self.__validate(dataset, architecture)
            if self.__is_checkpoint(valid_loss, best_loss):
                self.saver.save(PytorchModel(architecture), self.optimizer)
            if self.stopper.should_stop(valid_loss):
                self.saver.save(architecture, self.optimizer)
                break
            print("-" * 59)
            print('|\tEnd of Epoch {:3d} \t|\t Training Loss {:.4%} \t|\t Validation Loss {:.4%} \t|'.format(epoch + 1, train_loss, valid_loss))
            print("-" * 59)
        return best_loss, architecture

    def test(self, dataset: Dataset, model: Model) -> float:
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in self.dataset.test().batches():
                outputs = model(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        return loss / len(self.dataset.test().batches())

    def __train(self, dataset, architecture):
        loss = 0.
        architecture.train(True)
        for batch in dataset.train().batches():
            outputs = architecture(batch.inputs())
            loss += self.loss_function.compute(outputs, batch.targets())
            self.optimizer.move()
        architecture.train(False)
        return loss / len(dataset.train().batches())

    def __validate(self, dataset, architecture):
        loss = 0.
        architecture.eval()
        with torch.no_grad():
            for batch in dataset.valid().batches():
                outputs = architecture(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        return loss / len(dataset.valid().batches())

    def __is_checkpoint(self, loss, best_loss):
        return loss < best_loss