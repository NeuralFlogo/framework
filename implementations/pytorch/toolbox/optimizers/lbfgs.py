from typing import Iterator

from torch.nn import Parameter
from torch.optim import LBFGS

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer


class PytorchLbfgsOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, max_iterations: int, max_evaluation: int, tolerance_grad: float, tolerance_change: float, history_size: int):
        super(PytorchLbfgsOptimizer).__init__(learning_rate)
        self.optimizer = LBFGS(params=parameters, lr=learning_rate, max_iter=max_iterations, max_eval=max_evaluation, tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        self.optimizer.zero_grad()
