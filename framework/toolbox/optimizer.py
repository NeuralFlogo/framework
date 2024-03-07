from abc import ABC, abstractmethod

from framework.toolbox.scheduler import Scheduler


class Optimizer(ABC):
    def __init__(self, learning_rate: float, scheduler: Scheduler = None):
        self.scheduler = scheduler
        self.learning_rate = learning_rate

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def weights(self):
        pass
