from torch.nn import L1Loss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchMAELossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchMAELossFunction, self).__init__()
        self.loss = L1Loss()
