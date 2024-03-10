from abc import ABC, abstractmethod
from typing import List

from framework.toolbox.scheduler import Scheduler


class Optimizer(ABC):
    def __init__(self, learning_rate: float, schedulers: List[Scheduler] = None):
        self.schedulers = schedulers
        self.learning_rate = learning_rate

    @abstractmethod
    def move(self):
        pass

    @abstractmethod
    def weights(self):
        pass
