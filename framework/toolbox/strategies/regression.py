from abc import ABC

from framework.toolbox.loss import LossFunction
from framework.toolbox.strategy import Strategy


class RegressionStrategy(Strategy, ABC):
    def __init__(self, loss_function: LossFunction):
        self.loss_function = loss_function
