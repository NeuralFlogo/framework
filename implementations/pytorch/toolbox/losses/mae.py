from torch.nn import L1Loss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchMaeLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchMaeLossFunction, self).__init__()
        self.loss = L1Loss()
