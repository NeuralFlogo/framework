import numpy as np
import torch

from framework.toolbox.logger import Logger
from framework.toolbox.strategies.classification import ClassificationStrategy
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset


class PytorchClassificationStrategy(ClassificationStrategy):
    def evaluate(self, test_set: PytorchDataset, model: PytorchModel):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in test_set.batches():
                outputs = model(batch.inputs())
                predictions.extend(outputs.argmax(1).tolist())
                targets.extend(batch.targets().tolist())
        return self.__compute_accuracy(predictions, targets)

    def __compute_accuracy(self, predictions, targets) -> float:
        correct = len([1 for prediction, target in zip(predictions, targets) if prediction == target])
        return correct / len(predictions)

