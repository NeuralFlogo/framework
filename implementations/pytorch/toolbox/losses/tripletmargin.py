from torch.nn import TripletMarginLoss

from implementations.pytorch.toolbox.loss import PytorchLossFunction


class PytorchTripletMarginLossFunction(PytorchLossFunction):
    def __init__(self):
        super(PytorchTripletMarginLossFunction, self).__init__()
        self.loss = TripletMarginLoss()
