from abc import ABC
from typing import List

from framework.toolbox.optimizer import Optimizer
from implementations.pytorch.toolbox.scheduler import PytorchScheduler


class PytorchOptimizer(Optimizer, ABC):
    def __init__(self, learning_rate: float, schedulers: List[PytorchScheduler] = None):
        super(PytorchOptimizer, self).__init__(learning_rate=learning_rate, schedulers=schedulers)
