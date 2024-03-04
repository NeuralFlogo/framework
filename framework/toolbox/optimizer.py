from abc import ABC, abstractmethod

from framework.toolbox.scheduler import Scheduler


class Optimizer(ABC):
    def __init__(self, scheduler: Scheduler = None):
        self.scheduler = scheduler

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def weights(self):
        pass
