from framework.architecture.model import Model
from framework.toolbox.dataset import Dataset
from framework.toolbox.strategies.classification import ClassificationStrategy


class PytorchClassificationStrategy(ClassificationStrategy):
    def evaluate(self, test_set: Dataset, model: Model) -> float:
        pass