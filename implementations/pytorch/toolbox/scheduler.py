from abc import ABC, abstractmethod

from framework.toolbox.scheduler import Scheduler


class PytorchScheduler(Scheduler, ABC):
    @abstractmethod
    def init(self, optimizer):
        pass
