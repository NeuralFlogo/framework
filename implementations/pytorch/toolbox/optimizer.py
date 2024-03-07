from abc import abstractmethod, ABC

from framework.toolbox.optimizer import Optimizer
from implementations.pytorch.toolbox.scheduler import PytorchScheduler


class PytorchOptimizer(Optimizer, ABC):
    def __init__(self, learning_rate: float, scheduler: PytorchScheduler = None):
        super(PytorchOptimizer, self).__init__(learning_rate=learning_rate, scheduler=scheduler)
