from torch.nn import NLLLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchCrossEntropyLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchCrossEntropyLossFunction, self).__init__()
        self.loss = NLLLoss()
