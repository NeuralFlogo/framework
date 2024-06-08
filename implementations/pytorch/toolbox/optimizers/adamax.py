from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import Adamax

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchAdamaxOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        super(PytorchAdamaxOptimizer, self).__init__(learning_rate)
        self.optimizer = Adamax(params=parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        self.optimizer.zero_grad()
