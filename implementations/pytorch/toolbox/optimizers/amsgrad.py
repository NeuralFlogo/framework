from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import Adam

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchAmsgradOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        super(PytorchAmsgradOptimizer, self).__init__(learning_rate)
        self.optimizer = Adam(params=parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=True)
        self.optimizer.zero_grad()
        