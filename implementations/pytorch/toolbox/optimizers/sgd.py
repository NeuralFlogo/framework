from typing import Iterator

from torch.nn import Parameter
from torch.optim import SGD

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchSGDOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, momentum: float, dampening: float, weight_decay: float):
        super(PytorchSGDOptimizer, self).__init__(learning_rate)
        self.optimizer = SGD(params=parameters, lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay)
        self.optimizer.zero_grad()
