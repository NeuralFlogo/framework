import numpy as np
import torch

from framework.toolbox.logger import Logger
from framework.toolbox.strategies.classification import ClassificationStrategy
from implementations.pytorch.architecture.model import PytorchModel
from implementations.pytorch.toolbox.dataset import PytorchDataset


class PytorchClassificationStrategy(ClassificationStrategy):
    def evaluate(self, test_set: PytorchDataset, model: PytorchModel, architecture_name: str, experiment_name: str, logger: Logger):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in test_set.batches():
                outputs = model(batch.inputs())
                predictions.extend(outputs.argmax(1).tolist())
                targets.extend(batch.targets().tolist())
        logger.log_test(architecture_name, experiment_name, "accuracy", self.__compute_accuracy(np.array(predictions), np.array(targets)))
        return predictions, targets

    def __compute_accuracy(self, predictions, targets) -> float:
        return np.sum(np.all(predictions == targets, axis=1)) / len(predictions)

