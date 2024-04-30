from torch.nn import CTCLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchCTCLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchCTCLossFunction, self).__init__()
        self.loss = CTCLoss()
        