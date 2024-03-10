from typing import Union, List

from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):
    def compute(self, predictions: Tensor, targets: Tensor, training: bool = False) -> Union[float, List[float]]:
        pass
