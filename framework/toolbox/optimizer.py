from abc import ABC, abstractmethod


class Scheduler:
    pass


class Optimizer(ABC):
    def __init__(self, learning_rate: float, scheduler: Scheduler = None):
        self.learning_rate = learning_rate
        self.scheduler = scheduler

    @abstractmethod
    def move(self):
        pass
