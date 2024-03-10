from typing import Iterator, List

from torch.nn import Parameter
from torch.optim import SGD

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.scheduler import PytorchScheduler


class PytorchSGDOptimizer(PytorchOptimizer):
    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, momentum: float, schedulers: List[PytorchScheduler] = None):
        super(PytorchSGDOptimizer, self).__init__(learning_rate, schedulers)
        self.optimizer = SGD(params=parameters, lr=learning_rate, momentum=momentum)
        self.optimizer.zero_grad()
        self.init_schedulers()

    def move(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.refine_learning()

    def weights(self):
        return self.optimizer.state_dict()
