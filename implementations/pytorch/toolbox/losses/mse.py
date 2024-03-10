from typing import Union, List

from torch import Tensor
from torch.nn import MSELoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class MsePytorchLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchLossFunction, self).__init__()
        self.loss = MSELoss()

    def compute(self, predictions: Tensor, targets: Tensor, training: bool = False) -> Union[float, List[float]]:
        predictions = self.__check_predictions(predictions)
        loss = self.loss(predictions, targets)
        if training:
            loss.backward()
        return loss.item()
