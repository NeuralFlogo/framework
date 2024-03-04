from abc import abstractmethod
from typing import Union, List

from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):
    def compute(self, predictions, targets) -> Union[float, List[float]]:
        pass
