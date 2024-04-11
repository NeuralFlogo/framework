import torch

from framework.toolbox.logger import Logger
from framework.toolbox.strategies.regression import RegressionStrategy
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchRegressionStrategy(RegressionStrategy):
    def __init__(self, loss_function: PytorchLossFunction):
        super(PytorchRegressionStrategy, self).__init__(loss_function)

    def evaluate(self, test_set: PytorchDataset, model: PytorchModel, architecture_name: str, experiment_name: str, logger: Logger) -> float:
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in test_set.batches():
                outputs = model(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        logger.log_test(architecture_name, experiment_name, loss / len(test_set.batches()))
        return loss / len(test_set.batches())
