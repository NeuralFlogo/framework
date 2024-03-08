import torch

from framework.toolbox.strategies.regression import RegressionStrategy
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.datasets.datasets.pytorch_dataset import PytorchDataset
from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchRegressionStrategy(RegressionStrategy):
    def __init__(self, loss_function: PytorchLossFunction):
        super(PytorchRegressionStrategy, self).__init__(loss_function)

    def evaluate(self, test_set: PytorchDataset, model: PytorchModel) -> float:
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in test_set.batches():
                outputs = model(batch.inputs())
                loss += self.loss_function.compute(outputs, batch.targets())
        return loss / len(test_set.batches())
