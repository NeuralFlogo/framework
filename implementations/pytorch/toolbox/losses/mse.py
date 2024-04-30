from torch.nn import MSELoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchMseLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchMseLossFunction, self).__init__()
        self.loss = MSELoss()
