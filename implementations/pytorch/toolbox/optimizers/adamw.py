from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import AdamW

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchAdamWOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        super(PytorchAdamWOptimizer, self).__init__(learning_rate)
        self.optimizer = AdamW(params=parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        self.optimizer.zero_grad()
