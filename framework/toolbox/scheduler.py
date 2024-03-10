from abc import ABC, abstractmethod


class Scheduler(ABC):
    @abstractmethod
    def move(self):
        pass
