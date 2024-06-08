from torch.nn import MSELoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchMSELossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchMSELossFunction, self).__init__()
        self.loss = MSELoss()
