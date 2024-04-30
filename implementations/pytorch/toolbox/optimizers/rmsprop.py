from typing import Iterator

from torch.nn import Parameter
from torch.optim import RMSprop

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchRMSPropOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, alpha: float, eps: float, weight_decay: float, momentum: float):
        super(PytorchRMSPropOptimizer).__init__(learning_rate)
        self.optimizer = RMSprop(parameters, lr=learning_rate, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        self.optimizer.zero_grad()
