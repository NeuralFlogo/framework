import torch

from framework.toolbox.experiment import Experiment
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.datasets.datasets.pytorch_dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchSaver
from implementations.pytorch.toolbox.stopper import PytorchEarlyStopper


class PytorchExperiment(Experiment):
    def __init__(self, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, stopper: PytorchEarlyStopper,
                 saver: PytorchSaver):
        super().__init__(optimizer, loss_function, stopper, saver)

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset,
            architecture: PytorchArchitecture):
        best_loss = float("inf")
        for epoch in range(epochs):
            print("-" * 25, 'Epoch {}'.format(epoch + 1), "-" * 29)
            train_loss = self.__train(training_set, architecture)
            valid_loss = self.__validate(validation_set, architecture)
            if self.__is_checkpoint(valid_loss, best_loss):
                print("The model is improving from {} to {}.".format(best_loss, valid_loss))
                best_loss = valid_loss
            #     self.saver.save(PytorchModel(architecture), self.optimizer)
            # if self.stopper.should_stop(valid_loss):
            #     self.saver.save(PytorchModel(architecture), self.optimizer)
            #     return best_loss, PytorchModel(architecture)
            print("-" * 59)
            print('|\tEnd of Epoch {:3d} \t|\t Training Loss {:.3f} \t|\t Validation Loss {:.3f} \t|'
                  .format(epoch + 1, train_loss, valid_loss))
            print("-" * 59)
        return valid_loss, PytorchModel(architecture)

    def __train(self, dataset: PytorchDataset, architecture: PytorchArchitecture):
        loss = 0.
        architecture.train(True)
        for batch in dataset.batches():
            outputs = architecture(batch.inputs())
            loss += self.loss_function.compute(outputs, batch.targets(), True)
            self.optimizer.move()
        architecture.train(False)
        return loss / len(dataset.batches())

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
