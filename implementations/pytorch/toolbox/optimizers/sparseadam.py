from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import SparseAdam

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchSparseAdamOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float,):
        super(PytorchSparseAdamOptimizer, self).__init__(learning_rate)
        self.optimizer = SparseAdam(params=parameters, lr=learning_rate, betas=betas, eps=eps)
        self.optimizer.zero_grad()
        