from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adadelta

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchAdadeltaOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, rho: float = 0.9, eps: float = 1e-8, weight_decay:float = 0):
        super(PytorchAdadeltaOptimizer, self).__init__(learning_rate)
        self.optimizer = Adadelta(params=parameters, lr=learning_rate, rho=rho, eps=eps, weight_decay=weight_decay)
        self.optimizer.zero_grad()
