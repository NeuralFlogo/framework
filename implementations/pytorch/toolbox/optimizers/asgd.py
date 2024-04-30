from typing import Iterator

from torch.nn import Parameter
from torch.optim import ASGD

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchASGDOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, lambd: float, alpha: float, t0: float, weight_decay: float):
        super(PytorchASGDOptimizer).__init__(learning_rate)
        self.optimizer = ASGD(params=parameters, lr=learning_rate, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        self.optimizer.zero_grad()
