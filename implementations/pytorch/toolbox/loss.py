from abc import abstractmethod

from torch import Tensor

from framework.toolbox.loss import LossFunction


class PytorchLossFunction(LossFunction):
    @abstractmethod
    def compute(self, outputs: Tensor, targets: Tensor) -> Tensor:
        pass
