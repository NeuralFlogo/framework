from torch import nn

from implementations.pytorch.architecture.layers.regularization import PytorchRegularizationLayer


class PytorchDropoutLayer(PytorchRegularizationLayer):
    def __init__(self, probability: float):
        super(PytorchDropoutLayer, self).__init__()
        self.layer = nn.Dropout(p=probability)
