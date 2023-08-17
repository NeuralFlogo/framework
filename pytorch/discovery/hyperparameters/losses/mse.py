from torch import nn

from discovery.hyperparameters.loss import Loss


class PytorchMSELoss(Loss):
    def __init__(self):
        self.function = nn.MSELoss()

    def compute(self, predictions, labels):
        loss = self.function(predictions, labels)
        loss.backward()
        return loss.item()
