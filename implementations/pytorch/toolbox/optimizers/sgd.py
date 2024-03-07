from typing import Iterator

from torch.nn import Parameter
from torch.optim import SGD

from implementations.pytorch.toolbox.optimizer import PytorchOptimizer
from implementations.pytorch.toolbox.scheduler import PytorchScheduler


class SgdPytorchOptimizer(PytorchOptimizer):
    def weights(self):
        return self.optimizer.state_dict()

    def __init__(self, parameters: Iterator[Parameter], learning_rate: float, momentum: float, scheduler: PytorchScheduler = None):
        super(SgdPytorchOptimizer, self).__init__(learning_rate, scheduler)
        self.optimizer = SGD(params=parameters, lr=learning_rate, momentum=momentum)
        self.optimizer.zero_grad()

    def move(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
