from typing import Union, List

from torch import Tensor
from torch.nn import MSELoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class MsePytorchLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchLossFunction, self).__init__()
        self.loss = MSELoss()

    def compute(self, outputs: Tensor, targets: Tensor) -> Union[float, List[float]]:
        loss = self.loss(outputs, targets)
        loss.backward()
        return loss.item()