from abc import ABC, abstractmethod


class Batch(ABC):

    @abstractmethod
    def targets(self):
        pass

    @abstractmethod
    def inputs(self):   # Dado un batch, oobtener los inputs y los outputs
        pass