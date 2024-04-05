from abc import ABC

from framework.toolbox.optimizer import Optimizer


class PytorchOptimizer(Optimizer):
    def __init__(self, learning_rate: float):
        super(PytorchOptimizer, self).__init__(learning_rate=learning_rate)

    def move(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def weights(self):
        return self.optimizer.state_dict()
