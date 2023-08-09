from torch import nn

from framework.discovery.hyperparameters.loss import Loss


class CrossEntropyLoss(Loss):

    def __init__(self) -> None:
        self.function = nn.CrossEntropyLoss()

    def compute(self, predictions, labels):
        loss = self.function(predictions, labels)
        loss.backward()
        return loss.item()
