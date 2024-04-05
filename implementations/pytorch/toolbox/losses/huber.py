from torch.nn import HuberLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchHuberLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchHuberLossFunction, self).__init__()
        self.loss = HuberLoss()
