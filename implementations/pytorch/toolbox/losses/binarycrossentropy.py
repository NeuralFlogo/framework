from torch.nn import BCELoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchBinaryCrossEntropyLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchBinaryCrossEntropyLossFunction, self).__init__()
        self.loss = BCELoss()
