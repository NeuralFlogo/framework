from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import NAdam

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchNAdamOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float, weight_decay: float, momentum_decay: float):
        super(PytorchNAdamOptimizer, self).__init__(learning_rate)
        self.optimizer = NAdam(params=parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        self.optimizer.zero_grad()
