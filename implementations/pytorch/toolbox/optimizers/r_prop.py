from typing import Iterator, Tuple

from torch.nn import Parameter
from torch.optim import Rprop

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchRPropOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, etas: Tuple[float, float], step_sizes: Tuple[float, float]):
        super(PytorchRPropOptimizer).__init__(learning_rate)
        self.optimizer = Rprop(params=parameters, lr=learning_rate, etas=etas, step_sizes=step_sizes)
        self.optimizer.zero_grad()
