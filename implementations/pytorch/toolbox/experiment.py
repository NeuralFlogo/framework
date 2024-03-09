import torch

from framework.toolbox.experiment import Experiment
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.datasets.datasets.pytorch_dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.checkpoint_saver import PytorchCheckpointSaver
from implementations.pytorch.toolbox.result_saver import PytorchResultSaver
from implementations.pytorch.toolbox.stopper import PytorchEarlyStopper

BATCH_TO_SAVE = 10


class PytorchExperiment(Experiment):
    def __init__(self, name: str, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction,
                 stopper: PytorchEarlyStopper,
                 checkpoint_saver: PytorchCheckpointSaver):
        super().__init__(name, optimizer, loss_function, stopper, checkpoint_saver)

    def run(self, epochs: int, training_set: PytorchDataset, validation_set: PytorchDataset,
            architecture: PytorchArchitecture, result_saver: PytorchResultSaver):
        best_loss = float("inf")
        for epoch in range(epochs):
            train_loss = self.__train(training_set, architecture, epoch, result_saver)
            valid_loss = self.__validate(validation_set, architecture)
            if self.__is_checkpoint(valid_loss, best_loss):
                print("The model is improving from {} to {}.".format(best_loss, valid_loss))
                best_loss = valid_loss
            #     self.checkpoint_saver.save(PytorchModel(architecture), self.optimizer)
            # if self.stopper.should_stop(valid_loss):
            #     self.checkpoint_saver.save(PytorchModel(architecture), self.optimizer)
            #     return best_loss, PytorchModel(architecture)
            result_saver.save_epoch(architecture.name, self.name, epoch, train_loss, valid_loss)
        return valid_loss, PytorchModel(architecture)

    def __train(self, dataset: PytorchDataset, architecture: PytorchArchitecture, epoch, result_saver):
        loss = 0.
        previous_batch_loss = loss
        architecture.train(True)
        batches = dataset.batches()
        for n_batch in range(len(batches)):
            outputs = architecture(batches[n_batch].inputs())
            loss += self.loss_function.compute(outputs, batches[n_batch].targets(), True)
            if (n_batch + 1) % BATCH_TO_SAVE == 0:
                self.__save_batch(result_saver, architecture.name, epoch, n_batch,
                                  self.__get_batch_loss(loss, previous_batch_loss))
                previous_batch_loss = loss
            self.optimizer.move()
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

    def __save_batch(self, result_saver, architecture_name, epoch, n_batch, loss):
        result_saver.save_batch(architecture_name, self.name, epoch, n_batch + 1, loss)
