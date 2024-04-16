import torch

from framework.toolbox.strategies.regression import RegressionStrategy
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset
from implementations.pytorch.toolbox.device import PytorchDevice
from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchRegressionStrategy(RegressionStrategy):
    def __init__(self, loss_function: PytorchLossFunction):
        super(PytorchRegressionStrategy, self).__init__(loss_function)

    def evaluate(self, test_set: PytorchDataset, model: PytorchModel, device: PytorchDevice) -> float:
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in test_set.batches():
                outputs = model(batch.inputs().to(device.get()))
                loss += self.loss_function.validation_compute(outputs, batch.targets().to(device.get()))
        return loss / len(test_set.batches())
