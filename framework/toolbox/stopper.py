class EarlyStopper:
    def __init__(self, patience: int, delta: float):
        self.patience = patience
        self.delta = delta
        self.history = [float("inf")] * patience

    def should_stop(self, loss: float) -> bool:
        self.history.append(loss)
        return self.__oldest_loss() - self.__lowest_loss() < self.delta

    def __oldest_loss(self):
        return self.history.pop(0)

    def __lowest_loss(self):
        return min(self.history)
