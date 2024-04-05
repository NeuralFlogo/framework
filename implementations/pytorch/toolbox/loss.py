from typing import Union, List

import torch
from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):
    def compute(self, predictions: Tensor, targets: Tensor, training: bool = False) -> Union[float, List[float]]:
        predictions = self.__check_predictions(predictions)
        loss = self.loss(predictions, targets)
        if training:
            loss.backward()
        return loss.item()

    def __check_predictions(self, predictions: Tensor):
        if predictions.shape[1] == 1:
            return torch.squeeze(predictions, dim=1)
        return predictions
