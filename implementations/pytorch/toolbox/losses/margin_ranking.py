from torch.nn import MarginRankingLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchMarginRankingLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchMarginRankingLossFunction, self).__init__()
        self.loss = MarginRankingLoss()
