from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adagrad

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchAdagradOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, learning_rate_decay: float, eps: float, weight_decay: float):
        super(PytorchAdagradOptimizer, self).__init__(learning_rate)
        self.optimizer = Adagrad(params=parameters, lr=learning_rate, lr_decay= learning_rate_decay, eps=eps, weight_decay=weight_decay)
        self.optimizer.zero_grad()
