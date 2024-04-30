from torch.nn import CTCLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchCtcLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchCtcLossFunction, self).__init__()
        self.loss = CTCLoss()
        