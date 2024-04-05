from torch.nn import KLDivLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchKullbackLeiblerDivergenceLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchKullbackLeiblerDivergenceLossFunction, self).__init__()
        self.loss = KLDivLoss()
