from typing import Union, List

import torch
from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):

    def __check_predictions(self, predictions: Tensor):
        if predictions.shape[1] == 1:
            return torch.squeeze(predictions)
        return predictions

    def compute(self, predictions: Tensor, targets: Tensor, training: bool = False) -> Union[float, List[float]]:
        pass
