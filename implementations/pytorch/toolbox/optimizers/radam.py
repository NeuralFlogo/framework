from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import RAdam

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchRAdamOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        super(PytorchRAdamOptimizer, self).__init__(learning_rate)
        self.optimizer = RAdam(params=parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        self.optimizer.zero_grad()
