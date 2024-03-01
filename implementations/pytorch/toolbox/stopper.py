import numpy as np
from framework.toolbox.stopper import EarlyStopper


class PytorchEarlyStopper(EarlyStopper):
    def __init__(self, patience: int, delta: float):
        super(PytorchEarlyStopper, self).__init__(patience, delta)
        self.history = [np.Inf] * (patience + 1)

    def should_stop(self, loss) -> bool:
        self.history.append(loss)
        return self.__best_loss() - self.__oldest_loss() < self.delta

    def __oldest_loss(self):
        return self.history.pop(0)

    def __best_loss(self):
        return min(self.history)
