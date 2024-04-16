from typing import Union, List

import torch
from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):
    def training_compute(self, predictions: Tensor, targets: Tensor) -> Union[float, List[float]]:
        predictions = self.__check_predictions(predictions)
        loss = self.loss(predictions, targets)
        loss.backward()
        return loss.item()

    def validation_compute(self, predictions: Tensor, targets: Tensor) -> Union[float, List[float]]:
        predictions = self.__check_predictions(predictions)
        loss = self.loss(predictions, targets)
        return loss.item()

    def __check_predictions(self, predictions: Tensor):
        if predictions.shape[1] == 1:
            return torch.squeeze(predictions, dim=1)
        return predictions
