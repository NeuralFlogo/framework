from framework.toolbox.experiment import Experiment
from implementations.pytorch.architecture.architecture import PytorchArchitecture
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction
from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.saver import PytorchSaver
from implementations.pytorch.toolbox.stopper import PytorchEarlyStopper


class PytorchExperiment(Experiment):
    def __init__(self, optimizer: PytorchOptimizer, loss_function: PytorchLossFunction, stopper: PytorchEarlyStopper, saver: PytorchSaver):
        super().__init__(optimizer, loss_function, stopper, saver)

    def run(self, epochs: int, dataset: PytorchDataset, architecture: PytorchArchitecture):
        for epoch in range(epochs):
            total_loss = 0
            architecture.train(True)
            for batch in dataset.batches():
                outputs = architecture(batch.inputs())
                loss = self.loss_function.compute(outputs, batch.targets())
                self.optimizer.move()
                total_loss += loss
            if checker.is_checkpoint():
                self.saver.save(architecture)
            if self.stopper.should_stop(total_loss):
                self.saver.save(architecture, self.optimizer)